import dataclasses
import threading
import typing

import numpy as np
import neuromorphic_drivers as nd
import PySide6.QtCore
import PySide6.QtGraphs

from . import extension
from . import ui


def camera_thread_target(
    device: nd.prophesee_evk4.DeviceOptional,
    event_display: ui.EventDisplay,
    spectrum_line_series: PySide6.QtGraphs.QLineSeries,
):
    rpm_calculator = extension.RpmCalculator()
    spectrum = np.zeros(1000, dtype=np.float32)
    try:
        for status, packet in device:
            if packet is not None:
                if "dvs_events" in packet:
                    assert status.ring is not None and status.ring.current_t is not None
                    rpms = rpm_calculator.process(packet["dvs_events"], spectrum)
                    if rpms is not None:
                        spectrum_line_series.replace(
                            [
                                PySide6.QtCore.QPointF(index, spectrum[index])
                                for index in range(0, len(spectrum))
                            ]
                        )
                    event_display.push(
                        events=packet["dvs_events"], current_t=status.ring.current_t
                    )
                elif status.ring is not None and status.ring.current_t is not None:
                    event_display.push(
                        events=np.array([]), current_t=status.ring.current_t
                    )
    except (
        TypeError
    ):  # work-around for a bug in nd, fixed on main (awaiting next release)
        pass


def main():
    configuration = nd.prophesee_evk4.Configuration()
    device = nd.open(configuration=configuration, iterator_timeout=1.0 / 60.0)

    biases_names = set(dataclasses.asdict(configuration.biases).keys())

    def to_python(key: str, value: typing.Any):
        if key in biases_names:
            configuration.biases.__setattr__(key, value)
            device.update_configuration(configuration)
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
                            width: eventsView.width
                            height: eventsView.height
                            sensor_size: "{device.properties().width}x{device.properties().height}"
                            style: "exponential"
                            tau: 100000
                        }}
                    }}

                    GraphsView {{
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        Layout.minimumHeight: 400
                        Layout.maximumHeight: 400

                        axisX: ValueAxis {{
                            id: xAxis
                            min: 0.0
                            max: 1000.0
                        }}

                        axisY: ValueAxis {{
                            id: yAxis
                            min: 0.0
                            max: 1000.0
                        }}

                        LineSeries {{
                            objectName: "spectrum"
                            onPointsReplaced: {{

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
                                text: "Biases"
                                color: "#AAAAAA"
                            }}
                            {biases_qml}
                        }}
                    }}
                }}
            }}

            RoundButton {{
                text: "\u2731"
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
    camera_thread = threading.Thread(
        target=camera_thread_target,
        daemon=True,
        args=(device, event_display, spectrum_line_series),
    )
    camera_thread.start()
    app.run()
