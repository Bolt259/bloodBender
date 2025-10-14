from pathlib import Path
import json
import platformdirs
from typing import Optional


class PathPlugin:
    """Plugin to load / store files to an OS path location"""

    def __init__(self, path: Path, *, noload: bool = False, nostore: bool = False) -> None:
        self.path = path
        self.noload = noload
        self.nostore = nostore

    def load(self) -> Optional[dict]:
        if self.noload:
            return None
        
        try:
            with self.path.open() as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def update(self, token: dict) -> None:
        if self.nostore:
            return
        
        with self.path.open("w") as f:
            json.dump(token, f)


class OSCachedPlugin(PathPlugin):
    """Same as ``PathPlugin``, but saves/loads to the OS's user-cache directory (appdata, ~/.cache/..., etc)."""

    def __init__(
        self,
        appname: str,
        appauthor: str,
        version: Optional[str] = None,
        filename: str = "token.json",
        *, noload: bool = False, nostore: bool = False
    ) -> None:
        dirs = platformdirs.PlatformDirs(appname=appname, appauthor=appauthor, version=version)
        dir = dirs.user_cache_path
        dir.mkdir(parents=True, exist_ok=True)
        super().__init__(dir / filename, noload=noload, nostore=nostore)
