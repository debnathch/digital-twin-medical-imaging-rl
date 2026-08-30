import heapq
from typing import Optional, List, Tuple
from simulator.events.event import Event

class EventQueue:
    def __init__(self):
        self._queue: List[Tuple[float, int, int, Event]] = []
        self._entry_finder = {}
        self._REMOVED = '<removed-event>'
        self._counter = 0
        
    def schedule(self, event: Event):
        if event.event_id in self._entry_finder:
            self.cancel(event.event_id)
        count = self._counter
        self._counter += 1
        entry = [event.timestamp, event.priority, count, event]
        self._entry_finder[event.event_id] = entry
        heapq.heappush(self._queue, entry)
        
    def cancel(self, event_id: str):
        entry = self._entry_finder.pop(event_id, None)
        if entry is not None:
            entry[-1] = self._REMOVED
            
    def next_event(self) -> Optional[Event]:
        while self._queue:
            timestamp, priority, count, event = heapq.heappop(self._queue)
            if event is not self._REMOVED:
                del self._entry_finder[event.event_id]
                return event
        return None
        
    def peek(self) -> Optional[Event]:
        while self._queue:
            entry = self._queue[0]
            if entry[-1] is not self._REMOVED:
                return entry[-1]
            heapq.heappop(self._queue)
        return None
        
    def is_empty(self) -> bool:
        return self.peek() is None
        
    def __len__(self) -> int:
        return len(self._entry_finder)
