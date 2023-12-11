from typing import Type, FrozenSet
self.__allowed_events_in_packet: FrozenSet[Type[BinLogEvent]] = frozenset(
    [TableMapEvent, RotateEvent]).union(self.__allowed_events)
