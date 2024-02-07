from abc import ABC, abstractmethod
import asyncio
from typing import Optional, Dict, Any, Union


class EventTracker(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def user_agent(self) -> str:
        pass

    @user_agent.setter
    @abstractmethod
    def user_agent(self, value: str) -> None:
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        pass

    @domain.setter
    @abstractmethod
    def domain(self, value: str) -> None:
        pass

    @abstractmethod
    async def track_event(self, name: str, url: str, referrer: Optional[str] = None,
                          props: Optional[Dict[str, Any]] = None) -> str:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
