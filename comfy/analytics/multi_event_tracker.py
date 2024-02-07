import asyncio
from typing import List, Optional, Dict, Any, Union

from .event_tracker import EventTracker


class MultiEventTracker(EventTracker):
    def __init__(self, trackers: List[EventTracker]) -> None:
        super().__init__()
        self.trackers = trackers

    async def track_event(self, name: str, url: str, referrer: Optional[str] = None,
                          props: Optional[Dict[str, Any]] = None) -> None:
        tasks = [tracker.track_event(name, url, referrer, props) for tracker in self.trackers]
        await asyncio.gather(*tasks)

    async def close(self) -> None:
        tasks = [tracker.close() for tracker in self.trackers]
        await asyncio.gather(*tasks)

    @property
    def user_agent(self) -> str:
        return next(tracker.user_agent for tracker in self.trackers) if len(self.trackers) > 0 else "(unknown)"

    @user_agent.setter
    def user_agent(self, value: str) -> None:
        for tracker in self.trackers:
            tracker.user_agent = value

    @property
    def domain(self) -> str:
        return next(tracker.domain for tracker in self.trackers) if len(self.trackers) > 0 else ("unknown")

    @domain.setter
    def domain(self, value: str) -> None:
        for tracker in self.trackers:
            tracker.domain = value
