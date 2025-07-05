import typing

import numpy
import numpy.typing

class RpmCalculator:
    def __init__(self) -> None: ...
    def process(self, events: numpy.ndarray, spectrum: numpy.typing.NDArray[numpy.float32]) -> typing.Optional[list[float]]: ...
