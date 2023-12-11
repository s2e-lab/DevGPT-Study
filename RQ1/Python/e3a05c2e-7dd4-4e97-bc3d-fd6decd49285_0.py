class ManualLinearScaler:

    def __init__(self, data_min=0.0, data_max=1.0, scale_min=0.0, scale_max=1.0):
        self._data_min = data_min
        self._data_max = data_max
        self._data_range = self._data_max - self._data_min
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._scale_range = self._scale_max - self._scale_min

    def scale(self, value):
        normalized = (value - self._data_min) / self._data_range
        return normalized * self._scale_range + self._scale_min
