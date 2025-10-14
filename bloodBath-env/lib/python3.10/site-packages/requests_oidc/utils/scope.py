from typing import List, Optional


def make_scope(scope: Optional[List[str]]) -> List[str]:
    if scope is None:
        return ["openid"]

    if "openid" in scope:
        return scope

    return ["openid"] + scope
