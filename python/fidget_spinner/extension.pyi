import typing

import numpy
import numpy.typing

class RpmCalculator:
    def __init__(self) -> None: ...
    def process(
        self,
        events: numpy.ndarray,
        spectrum: numpy.typing.NDArray[numpy.float32],
        autocorrelation: numpy.typing.NDArray[numpy.float32],
        autocorrelation_detections: numpy.typing.NDArray[numpy.float32],
        amplitude_threshold: float,
        autocorrelation_threshold: float,
        frequency_multiplier: float,
    ) -> typing.Optional[list[float]]: ...
