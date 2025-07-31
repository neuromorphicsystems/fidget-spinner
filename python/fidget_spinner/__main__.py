import dataclasses
import threading
import typing

import numpy as np
import neuromorphic_drivers as nd
import PySide6.QtCore
import PySide6.QtGraphs
import scipy.ndimage

from . import extension
from . import ui

FFT_FREQUENCY: float = 512.0  # must be the same as FFT_FREQUENCY in src/lib.rs
FFT_SAMPLES: int = 1024  # must be the same as FFT_SAMPLES in src/lib.rs
SAMPLING_FREQUENCY: float = 10.0
# must be the same as SAMPLING_FREQUENCY in src/lib.rs

MINIMUM_FILTER_SIZE: int = 5
DEFAULT_MAXIMUM_LATENCY: int = 3000
DEFAULT_AMPLITUDE_THRESHOLD: int = 100
DEFAULT_AUTOCORRELATION_THRESHOLD: int = 40
DEFAULT_FREQUENCY_DIVIDER: int = 3


@dataclasses.dataclass
class AlgorithmConfiguration:
    maximum_latency_ms: int
    amplitude_threshold: int
    autocorrelation_threshold: int
    frequency_divider: int
    pause: bool


@dataclasses.dataclass
class Cursors:
    spectrum: float
    autocorrelation: float
    rpms: float
    changed: bool


def camera_thread_target(
    algorithm_configuration: AlgorithmConfiguration,
    cursors: Cursors,
    device: nd.prophesee_evk4.PropheseeEvk4DeviceOptional,
    event_display: ui.EventDisplay,
    spectrum_line_series: PySide6.QtGraphs.QLineSeries,
    spectrum_amplitude_threshold_line_series: PySide6.QtGraphs.QLineSeries,
    autocorrelation_line_series: PySide6.QtGraphs.QLineSeries,
    autocorrelation_threshold_line_series: PySide6.QtGraphs.QLineSeries,
    autocorrelation_detection_line_series: PySide6.QtGraphs.QLineSeries,
    rpms_line_series: PySide6.QtGraphs.QLineSeries,
):
    rpm_calculator = extension.RpmCalculator()
    spectrum = np.zeros(FFT_SAMPLES // 2, dtype=np.float32)
    autocorrelation = np.zeros(FFT_SAMPLES // 2, dtype=np.float32)
    autocorrelation_detections = np.zeros(4, dtype=np.float32)
    rpms = np.zeros(300, dtype=np.float32)  # 30 s @ 10 Hz
    filtered_rpms = np.zeros(len(rpms), dtype=np.float32)
    first = True
    for status, packet in device:
        delay = status.delay()
        if (
            delay is not None
            and delay * 1000.0 > algorithm_configuration.maximum_latency_ms
        ):
            device.clear_backlog(0)
        if first or not algorithm_configuration.pause:
            if packet is not None:
                if packet.polarity_events is not None:
                    assert status.ring is not None and status.ring.current_t is not None
                    new_rpms = rpm_calculator.process(
                        packet.polarity_events,
                        spectrum,
                        autocorrelation,
                        autocorrelation_detections,
                        algorithm_configuration.amplitude_threshold / 10.0,
                        algorithm_configuration.autocorrelation_threshold / 100.0,
                        1.0 / algorithm_configuration.frequency_divider,
                    )
                    if first or new_rpms is not None:
                        first = False
                        cursors.changed = True
                        spectrum_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(
                                    index * FFT_FREQUENCY / FFT_SAMPLES,
                                    spectrum[index],
                                )
                                for index in range(0, len(spectrum))
                            ]
                        )
                        spectrum_line_series.setProperty(
                            "maximum", float(spectrum.max())
                        )
                        spectrum_amplitude_threshold_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(
                                    0.0,
                                    algorithm_configuration.amplitude_threshold / 10.0,
                                ),
                                PySide6.QtCore.QPointF(
                                    FFT_FREQUENCY / 2,
                                    algorithm_configuration.amplitude_threshold / 10.0,
                                ),
                            ]
                        )
                        autocorrelation_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(
                                    index * FFT_FREQUENCY / FFT_SAMPLES,
                                    autocorrelation[index],
                                )
                                for index in range(0, len(autocorrelation))
                            ]
                        )
                        autocorrelation_threshold_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(
                                    0.0,
                                    algorithm_configuration.autocorrelation_threshold
                                    / 100.0,
                                ),
                                PySide6.QtCore.QPointF(
                                    FFT_FREQUENCY / 2,
                                    algorithm_configuration.autocorrelation_threshold
                                    / 100.0,
                                ),
                            ]
                        )
                        if autocorrelation_detections[2] >= 0.0:
                            autocorrelation_detection_line_series.replace(
                                [
                                    PySide6.QtCore.QPointF(
                                        autocorrelation_detections[2],
                                        -0.5,
                                    ),
                                    PySide6.QtCore.QPointF(
                                        autocorrelation_detections[2],
                                        autocorrelation_detections[3],
                                    ),
                                ]
                            )
                        else:
                            autocorrelation_detection_line_series.replace([])
                        if new_rpms is not None:
                            rpms[0 : -len(new_rpms)] = rpms[len(new_rpms) :]
                            rpms[-len(new_rpms) :] = new_rpms
                        scipy.ndimage.minimum_filter(
                            rpms,
                            size=MINIMUM_FILTER_SIZE,
                            output=filtered_rpms,
                            mode="nearest",
                        )
                        rpms_line_series.setProperty(
                            "maximum", max(float(filtered_rpms.max()), 1.0)
                        )
                        rpms_line_series.setProperty(
                            "current", float(filtered_rpms[-1])
                        )
                        rpms_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(
                                    index / SAMPLING_FREQUENCY, filtered_rpms[index]
                                )
                                for index in range(0, len(filtered_rpms))
                            ]
                        )
                    event_display.push(
                        events=packet.polarity_events, current_t=status.ring.current_t
                    )
                elif status.ring is not None and status.ring.current_t is not None:
                    event_display.push(
                        events=np.array([]), current_t=status.ring.current_t
                    )
        if cursors.changed:
            spectrum_line_series.setProperty(
                "cursor",
                float(spectrum[int(round((len(spectrum) - 1) * cursors.spectrum))]),
            )
            autocorrelation_line_series.setProperty(
                "cursor",
                float(
                    autocorrelation[
                        int(round((len(autocorrelation) - 1) * cursors.autocorrelation))
                    ]
                ),
            )
            rpms_line_series.setProperty(
                "cursor",
                float(rpms[int(round((len(rpms) - 1) * cursors.rpms))]),
            )
            cursors.changed = False


def main():
    configuration = nd.prophesee_evk4.Configuration()
    algorithm_configuration = AlgorithmConfiguration(
        maximum_latency_ms=DEFAULT_MAXIMUM_LATENCY,
        amplitude_threshold=DEFAULT_AMPLITUDE_THRESHOLD,
        autocorrelation_threshold=DEFAULT_AUTOCORRELATION_THRESHOLD,
        frequency_divider=DEFAULT_FREQUENCY_DIVIDER,
        pause=False,
    )
    cursors = Cursors(
        spectrum=0.0,
        autocorrelation=0.0,
        rpms=0.0,
        changed=False,
    )
    device = nd.open(configuration=configuration, iterator_timeout=1.0 / 60.0)

    biases_names = set(dataclasses.asdict(configuration.biases).keys())

    def to_python(key: str, value: typing.Any):
        if key in biases_names:
            configuration.biases.__setattr__(key, value)
            device.update_configuration(configuration)
        elif key == "maximum_latency_ms":
            algorithm_configuration.maximum_latency_ms = value
        elif key == "amplitude_threshold":
            algorithm_configuration.amplitude_threshold = value
        elif key == "autocorrelation_threshold":
            algorithm_configuration.autocorrelation_threshold = value
        elif key == "frequency_divider":
            algorithm_configuration.frequency_divider = value
        elif key == "pause":
            algorithm_configuration.pause = value
        elif key == "spectrum_cursor":
            if isinstance(value, (int, float)) and value >= 0.0 and value <= 1.0:
                cursors.spectrum = value
                cursors.changed = True
        elif key == "autocorrelation_cursor":
            if isinstance(value, (int, float)) and value >= 0.0 and value <= 1.0:
                cursors.autocorrelation = value
                cursors.changed = True
        elif key == "rpms_cursor":
            if isinstance(value, (int, float)) and value >= 0.0 and value <= 1.0:
                cursors.rpms = value
                cursors.changed = True
        else:
            print(f"Unknown to_python key {key} with value {value}")

    biases_qml = ""
    for name, value in dataclasses.asdict(configuration.biases).items():
        biases_qml += f"""
        RowLayout {{
            Layout.topMargin: 5
            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
            spacing: 20
            SpinBox {{
                palette.button: "#393939"
                palette.buttonText: "#FFFFFF"
                palette.text: "#FFFFFF"
                palette.base: "#191919"
                palette.mid: "#494949"
                palette.highlight: "#1E88E5"
                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                from: 0
                to: 255
                editable: true
                value: {value}
                onValueModified: {{
                    to_python.{name} = value;
                }}
            }}
            Text {{
                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                text: "{name}"
                color: "#CCCCCC"
            }}
        }}
        """

    app = ui.App(
        f"""
        import QtGraphs
        import QtQuick
        import QtQuick.Layouts
        import QtQuick.Controls
        import NeuromorphicDrivers

        Window {{
            id: window
            width: {device.properties().width}
            height: {device.properties().height}
            color: "#191919"
            property var showMenu: false

            RowLayout {{
                spacing: 0
                width: window.width
                height: window.height

                ColumnLayout {{
                    Rectangle {{
                        id: eventsView
                        Layout.alignment: Qt.AlignCenter
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        color: "transparent"

                        EventDisplay {{
                            id: eventDisplay
                            width: eventsView.width
                            height: eventsView.height
                            sensor_size: "{device.properties().width}x{device.properties().height}"
                            style: "exponential"
                            tau: 100000
                        }}
                    }}
                    RowLayout {{
                        Item {{
                            Layout.fillHeight: true
                            Layout.fillWidth: true
                            Layout.minimumHeight: 200
                            Layout.maximumHeight: 200

                            GraphsView {{
                                id: spectrumGraphsView
                                anchors.fill: parent
                                marginTop: 40
                                marginLeft: 30
                                marginRight: 30
                                marginBottom: 20
                                theme: GraphsTheme {{
                                    colorScheme: GraphsTheme.ColorScheme.Dark
                                    gridVisible: false
                                    seriesColors: ["#005AA0", "#888888"]
                                }}

                                LineSeries {{
                                    id: spectrum
                                    objectName: "spectrum"
                                    property var maximum: 1.0
                                    property var cursor: 0.0
                                    hoverable: true
                                }}

                                LineSeries {{
                                    objectName: "spectrum_amplitude_threshold"
                                }}

                                axisX: ValueAxis {{
                                    min: 0.0
                                    max: {FFT_FREQUENCY / 2}
                                    tickInterval: 100.0
                                }}

                                axisY: ValueAxis {{
                                    min: 0.0
                                    max: Math.max(spectrum.maximum, amplitude_threshold.value / 10.0) * 1.1
                                    tickInterval: Math.max(spectrum.maximum, amplitude_threshold.value / 10.0) * 1.1 / 2
                                }}
                            }}

                            Text {{
                                text: "Spectrum"
                                color: "#AAAAAA"
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: 10
                                anchors.leftMargin: 10
                            }}

                            Text {{
                                text: "Amplitude"
                                color: "#AAAAAA"
                                rotation: -90
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.left: parent.left
                            }}

                            Text {{
                                text: "Frequency (Hz)"
                                color: "#999999"
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                                anchors.topMargin: 10
                                anchors.leftMargin: 10
                            }}

                            MouseArea {{
                                id: spectrumPlotArea
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: spectrumGraphsView.plotArea.top
                                anchors.leftMargin: spectrumGraphsView.plotArea.left
                                width: spectrumGraphsView.plotArea.width
                                height: spectrumGraphsView.plotArea.height
                                hoverEnabled: true
                                onEntered: {{
                                    spectrumCursor.visible = true;
                                }}
                                onExited: {{
                                    spectrumCursor.visible = false;
                                }}
                            }}

                            RowLayout {{
                                property var value: spectrumPlotArea.mouseX / spectrumGraphsView.plotArea.width
                                id: spectrumCursor
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: spectrumGraphsView.plotArea.top
                                anchors.leftMargin: spectrumGraphsView.plotArea.left + spectrumPlotArea.mouseX - width + 1
                                height: spectrumGraphsView.plotArea.height
                                layoutDirection: Qt.RightToLeft
                                visible: false

                                onValueChanged: {{
                                    to_python.spectrum_cursor = value;
                                }}

                                Rectangle {{
                                    Layout.fillHeight: true
                                    Layout.minimumWidth: 2
                                    Layout.maximumWidth: 2
                                    color: "#FFFFFF"
                                }}

                                Rectangle {{
                                    id: spectrumCursorLabel
                                    color: "#AA000000"
                                    radius: 4
                                    width: spectrumCursorText.width + 8
                                    height: spectrumCursorText.height + 8
                                    implicitWidth: spectrumCursorText.width + 8
                                    implicitHeight: spectrumCursorText.height + 8
                                    Text {{
                                        id: spectrumCursorText
                                        anchors.centerIn: parent
                                        text: spectrum.cursor.toFixed(1)
                                        color: "#FFFFFF"
                                        font.pixelSize: 12
                                    }}
                                }}
                            }}
                        }}

                        Item {{
                            Layout.fillHeight: true
                            Layout.fillWidth: true
                            Layout.minimumHeight: 200
                            Layout.maximumHeight: 200

                            GraphsView {{
                                id: autocorrelationGraphsView
                                anchors.fill: parent
                                marginTop: 40
                                marginLeft: 30
                                marginRight: 30
                                marginBottom: 20
                                theme: GraphsTheme {{
                                    colorScheme: GraphsTheme.ColorScheme.Dark
                                    gridVisible: false
                                    seriesColors: ["#0082A9", "#888888", "#F3E9B3"]
                                }}

                                LineSeries {{
                                    id: autocorrelation
                                    objectName: "autocorrelation"
                                    property var maximum: 1.0
                                    property var cursor: 0.0
                                }}

                                LineSeries {{
                                    objectName: "autocorrelation_threshold"
                                }}

                                LineSeries {{
                                    objectName: "autocorrelation_detection"
                                }}

                                axisX: ValueAxis {{
                                    min: 0.0
                                    max: {FFT_FREQUENCY / 2}
                                    tickInterval: 100.0
                                }}

                                axisY: ValueAxis {{
                                    min: -0.5
                                    max: 1.0
                                    tickInterval: 0.5
                                }}
                            }}

                            Text {{
                                text: "Autocorrelation"
                                color: "#999999"
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: 10
                                anchors.leftMargin: 10
                            }}

                            Text {{
                                text: "Amplitude"
                                color: "#AAAAAA"
                                rotation: -90
                                anchors.verticalCenter: parent.verticalCenter
                            }}

                            Text {{
                                text: "Frequency (Hz)"
                                color: "#AAAAAA"
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                            }}

                            MouseArea {{
                                id: autocorrelationPlotArea
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: autocorrelationGraphsView.plotArea.top
                                anchors.leftMargin: autocorrelationGraphsView.plotArea.left
                                width: autocorrelationGraphsView.plotArea.width
                                height: autocorrelationGraphsView.plotArea.height
                                hoverEnabled: true
                                onEntered: {{
                                    autocorrelationCursor.visible = true;
                                }}
                                onExited: {{
                                    autocorrelationCursor.visible = false;
                                }}
                            }}

                            RowLayout {{
                                property var value: autocorrelationPlotArea.mouseX / autocorrelationGraphsView.plotArea.width
                                id: autocorrelationCursor
                                anchors.top: parent.top
                                anchors.left: parent.left
                                anchors.topMargin: autocorrelationGraphsView.plotArea.top
                                anchors.leftMargin: autocorrelationGraphsView.plotArea.left + autocorrelationPlotArea.mouseX - width + 1
                                height: autocorrelationGraphsView.plotArea.height
                                layoutDirection: Qt.RightToLeft
                                visible: false

                                onValueChanged: {{
                                    to_python.autocorrelation_cursor = value;
                                }}

                                Rectangle {{
                                    Layout.fillHeight: true
                                    Layout.minimumWidth: 2
                                    Layout.maximumWidth: 2
                                    color: "#FFFFFF"
                                }}

                                Rectangle {{
                                    id: autocorrelationCursorLabel
                                    color: "#AA000000"
                                    radius: 4
                                    width: autocorrelationCursorText.width + 8
                                    height: autocorrelationCursorText.height + 8
                                    implicitWidth: autocorrelationCursorText.width + 8
                                    implicitHeight: autocorrelationCursorText.height + 8
                                    Text {{
                                        id: autocorrelationCursorText
                                        anchors.centerIn: parent
                                        text: autocorrelation.cursor.toFixed(2)
                                        color: "#FFFFFF"
                                        font.pixelSize: 12
                                    }}
                                }}
                            }}
                        }}
                    }}
                    Item {{
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        Layout.minimumHeight: 200
                        Layout.maximumHeight: 200

                        GraphsView {{
                            id: rpmsGraphsView
                            anchors.fill: parent
                            marginTop: 40
                            marginLeft: 35
                            marginRight: 30
                            marginBottom: 20
                            theme: GraphsTheme {{
                                colorScheme: GraphsTheme.ColorScheme.Dark
                                gridVisible: false
                                seriesColors: ["#F3E9B3"]
                            }}

                            LineSeries {{
                                id: rpms
                                objectName: "rpms"
                                property var maximum: 1.0
                                property var current: 0.0
                                property var cursor: 0.0
                            }}

                            axisX: ValueAxis {{
                                min: 0.0
                                max: 30.0
                                tickInterval: 5.0
                            }}

                            axisY: ValueAxis {{
                                min: 0.0
                                max: rpms.maximum * 1.1
                                tickInterval: rpms.maximum * 1.1 / 2
                            }}
                        }}

                        Text {{
                            text: `Angular speed    ${{rpms.current}} rpm`
                            color: "#AAAAAA"
                            anchors.top: parent.top
                            anchors.left: parent.left
                            anchors.topMargin: 10
                            anchors.leftMargin: 10
                        }}

                        Text {{
                            text: "Rotations per minute"
                            color: "#AAAAAA"
                            rotation: -90
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: -35
                        }}

                        Text {{
                            text: "Time (s)"
                            color: "#999999"
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                            anchors.topMargin: 10
                            anchors.leftMargin: 10
                        }}

                        MouseArea {{
                            id: rpmsPlotArea
                            anchors.top: parent.top
                            anchors.left: parent.left
                            anchors.topMargin: rpmsGraphsView.plotArea.top
                            anchors.leftMargin: rpmsGraphsView.plotArea.left
                            width: rpmsGraphsView.plotArea.width
                            height: rpmsGraphsView.plotArea.height
                            hoverEnabled: true
                            onEntered: {{
                                rpmsCursor.visible = true;
                            }}
                            onExited: {{
                                rpmsCursor.visible = false;
                            }}
                        }}

                        RowLayout {{
                            property var value: rpmsPlotArea.mouseX / rpmsGraphsView.plotArea.width
                            id: rpmsCursor
                            anchors.top: parent.top
                            anchors.left: parent.left
                            anchors.topMargin: rpmsGraphsView.plotArea.top
                            anchors.leftMargin: rpmsGraphsView.plotArea.left + rpmsPlotArea.mouseX - width + 1
                            height: rpmsGraphsView.plotArea.height
                            layoutDirection: Qt.RightToLeft
                            visible: false

                            onValueChanged: {{
                                to_python.rpms_cursor = value;
                            }}

                            Rectangle {{
                                Layout.fillHeight: true
                                Layout.minimumWidth: 2
                                Layout.maximumWidth: 2
                                color: "#FFFFFF"
                            }}

                            Rectangle {{
                                id: rpmsCursorLabel
                                color: "#AA000000"
                                radius: 4
                                width: rpmsCursorText.width + 8
                                height: rpmsCursorText.height + 8
                                implicitWidth: rpmsCursorText.width + 8
                                implicitHeight: rpmsCursorText.height + 8
                                Text {{
                                    id: rpmsCursorText
                                    anchors.centerIn: parent
                                    text: `${{rpms.cursor.toFixed(0)}} rpms`
                                    color: "#FFFFFF"
                                    font.pixelSize: 12
                                }}
                            }}
                        }}
                    }}
                }}

                Rectangle {{
                    id: menu
                    Layout.alignment: Qt.AlignCenter
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                    Layout.minimumWidth: 300
                    Layout.maximumWidth: 300
                    visible: showMenu
                    color: "#292929"

                    ScrollView {{
                        y: 60
                        width: menu.width
                        height: menu.height - 60
                        clip: true
                        leftPadding: 20
                        rightPadding: 20
                        bottomPadding: 20
                        ColumnLayout {{
                            Text {{
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                text: "Display"
                                color: "#AAAAAA"
                            }}

                            Button {{
                                property var pause: false;
                                text: pause ? "Resume" : "Pause"
                                palette.button: "#393939"
                                palette.buttonText: "#FFFFFF"
                                onClicked: {{
                                    pause = !pause;
                                    to_python.pause = pause;
                                }}
                            }}

                            RowLayout {{
                                Layout.topMargin: 5
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                spacing: 20
                                SpinBox {{
                                    palette.button: "#393939"
                                    palette.buttonText: "#FFFFFF"
                                    palette.text: "#FFFFFF"
                                    palette.base: "#191919"
                                    palette.mid: "#494949"
                                    palette.highlight: "#1E88E5"
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    from: 1
                                    to: 100000
                                    editable: true
                                    value: 100
                                    onValueModified: {{
                                        eventDisplay.tau = value * 1000;
                                    }}
                                }}
                                Text {{
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    text: "𝜏 (ms)"
                                    color: "#CCCCCC"
                                }}
                            }}
                            Text {{
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                text: "Biases"
                                color: "#AAAAAA"
                            }}
                            {biases_qml}

                            Text {{
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                text: "Algorithm"
                                color: "#AAAAAA"
                            }}
                            RowLayout {{
                                Layout.topMargin: 5
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                spacing: 20
                                SpinBox {{
                                    palette.button: "#393939"
                                    palette.buttonText: "#FFFFFF"
                                    palette.text: "#FFFFFF"
                                    palette.base: "#191919"
                                    palette.mid: "#494949"
                                    palette.highlight: "#1E88E5"
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    from: 20
                                    to: 60000
                                    editable: true
                                    value: {algorithm_configuration.maximum_latency_ms}
                                    onValueModified: {{
                                        to_python.maximum_latency_ms = value
                                    }}
                                }}
                                Text {{
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    text: "Max latency (ms)"
                                    color: "#CCCCCC"
                                }}
                            }}
                            RowLayout {{
                                Layout.topMargin: 5
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                spacing: 20
                                SpinBox {{
                                    id: amplitude_threshold
                                    palette.button: "#393939"
                                    palette.buttonText: "#FFFFFF"
                                    palette.text: "#FFFFFF"
                                    palette.base: "#191919"
                                    palette.mid: "#494949"
                                    palette.highlight: "#1E88E5"
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    from: 0
                                    to: 1000
                                    editable: true
                                    value: {algorithm_configuration.amplitude_threshold}
                                    onValueModified: {{
                                        to_python.amplitude_threshold = value
                                    }}
                                }}
                                Text {{
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    text: "Amplitude thr"
                                    color: "#CCCCCC"
                                }}
                            }}
                            RowLayout {{
                                Layout.topMargin: 5
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                spacing: 20
                                SpinBox {{
                                    palette.button: "#393939"
                                    palette.buttonText: "#FFFFFF"
                                    palette.text: "#FFFFFF"
                                    palette.base: "#191919"
                                    palette.mid: "#494949"
                                    palette.highlight: "#1E88E5"
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    from: 0
                                    to: 100
                                    editable: true
                                    value: {algorithm_configuration.autocorrelation_threshold}
                                    onValueModified: {{
                                        to_python.autocorrelation_threshold = value
                                    }}
                                }}
                                Text {{
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    text: "Autocorrelation thr"
                                    color: "#CCCCCC"
                                }}
                            }}
                            RowLayout {{
                                Layout.topMargin: 5
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                spacing: 20
                                SpinBox {{
                                    id: frequency_divider
                                    palette.button: "#393939"
                                    palette.buttonText: "#FFFFFF"
                                    palette.text: "#FFFFFF"
                                    palette.base: "#191919"
                                    palette.mid: "#494949"
                                    palette.highlight: "#1E88E5"
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    from: 1
                                    to: 1000
                                    editable: true
                                    value: {algorithm_configuration.frequency_divider}
                                    onValueModified: {{
                                        to_python.frequency_divider = value
                                    }}
                                }}
                                Text {{
                                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                    text: "Frequency div"
                                    color: "#CCCCCC"
                                }}
                            }}
                        }}
                    }}
                }}
            }}

            RoundButton {{
                text: "···"
                palette.button: "#393939"
                palette.buttonText: "#FFFFFF"
                x: window.width - 50
                y: 10
                icon.width: width
                icon.height: height
                onClicked: {{
                    showMenu = !showMenu
                }}
            }}
        }}
        """,
        from_python_defaults={},
        to_python=to_python,
    )
    event_display = app.event_display()
    spectrum_line_series = app.line_series("spectrum")
    spectrum_amplitude_threshold_line_series = app.line_series(
        "spectrum_amplitude_threshold"
    )
    autocorrelation_line_series = app.line_series("autocorrelation")
    autocorrelation_threshold_line_series = app.line_series("autocorrelation_threshold")
    autocorrelation_detection_line_series = app.line_series("autocorrelation_detection")
    rpms_line_series = app.line_series("rpms")
    camera_thread = threading.Thread(
        target=camera_thread_target,
        daemon=True,
        args=(
            algorithm_configuration,
            cursors,
            device,
            event_display,
            spectrum_line_series,
            spectrum_amplitude_threshold_line_series,
            autocorrelation_line_series,
            autocorrelation_threshold_line_series,
            autocorrelation_detection_line_series,
            rpms_line_series,
        ),
    )
    camera_thread.start()
    app.run()
