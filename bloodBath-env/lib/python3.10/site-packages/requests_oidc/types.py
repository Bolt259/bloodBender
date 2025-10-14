from typing import Protocol, Optional


class Plugin(Protocol):
    def load(self) -> Optional[dict]:
        ...

    def update(self, token: dict) -> None:
        ...
