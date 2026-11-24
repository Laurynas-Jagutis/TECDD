from datetime import datetime


class Event:
    def __init__(self, case_id: str, activity: str, timestamp: str, global_event_index: int | None = None):
        self.case_id = case_id
        self.activity = activity
        self.timestamp_str = timestamp
        self.global_event_index = global_event_index
        self.timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
