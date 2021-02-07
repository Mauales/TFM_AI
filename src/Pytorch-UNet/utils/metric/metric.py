from abc import ABC, abstractmethod

class Metric(ABC):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add(self):
        pass
