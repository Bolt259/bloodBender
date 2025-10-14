import time
from typing import List


def access_expired(token: dict, margin: int = 0) -> bool:
    return (time.time()) > token["expires_at"] + margin


def refresh_expired(token: dict, margin: int = 0) -> bool:
    if "offline_access" in token['scope']:
        return False

    refresh_expires_at = (
        token["refresh_expires_in"] - token["expires_in"] + token["expires_at"]
    )
    return (time.time() + margin) > (refresh_expires_at + margin)

def scope_mismatch(token: dict, scopes: List[str]) -> bool:
    return any(scope not in token['scope'] for scope in scopes)
