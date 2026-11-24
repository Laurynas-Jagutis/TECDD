from collections import OrderedDict

from event import Event


class CaseBuffer:
    """The last K events pf every running case."""
    
    def __init__(self, prefix_len, max_idle_events=25000):
        """max_idle_events: evict a case once more than this many events have been added since its last event"""
        self.prefix_len = prefix_len
        self.max_idle_events = max_idle_events
        
        self.buffer: OrderedDict[str, list[Event]] = OrderedDict()
        self.last_seen: dict[str, int] = {}
        self.events_seen = 0
        
    def add(self, event: Event):
        self.events_seen += 1
        case_id = event.case_id
        if case_id in self.buffer:
            self.buffer.move_to_end(case_id)
        else:
            self.buffer[case_id] = []
        events = self.buffer[case_id]
        events.append(event)
        if len(events) > self.prefix_len:
            events.pop(0)
        self.last_seen[case_id] = self.events_seen
        self._evict_idle()
        
    def _evict_idle(self):
        while self.buffer:
            # get the case that has been inactive the longest
            oldest = next(iter(self.buffer))
            if self.events_seen - self.last_seen[oldest] <= self.max_idle_events:
                break
            del self.buffer[oldest]
            del self.last_seen[oldest]
            
    def remove(self, case_id: str):
        self.buffer.pop(case_id, None)
        self.last_seen.pop(case_id, None)
            
    def get_prefix_for_a_case(self, case_id: str):
        events = self.buffer.get(case_id, [])
        activities = [event.activity for event in events]
        time_deltas = [0.0] + [ 
            (events[i].timestamp - events[i -1].timestamp).total_seconds() for i in range(1, len(events))
        ]
        return activities, time_deltas[:len(events)]
