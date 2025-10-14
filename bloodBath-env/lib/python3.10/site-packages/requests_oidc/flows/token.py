from typing import List, Optional

from requests_oauthlib import OAuth2Session  # type: ignore

from ..types import Plugin
from ..utils import ServerDetails, make_scope
from .utils import refresh_expired, scope_mismatch


def make_token_session(
    oidc_url: str,
    client_id: str,
    scope: Optional[List[str]] = None,
    *,
    klass=OAuth2Session,
    plugin: Optional[Plugin] = None,
    **kwargs,
) -> OAuth2Session:
    auth_server = ServerDetails.discover(oidc_url)
    scope = make_scope(scope)

    def updater(token: dict) -> None:
        if plugin:
            plugin.update(token)

    token = None if plugin is None else plugin.load()
    if token is None:
        raise RuntimeError("Token not provided!")
    elif refresh_expired(token, margin=15):
        raise RuntimeError("Token is expired!")
    elif  scope_mismatch(token, scope):
        raise RuntimeError("Token scope is wrong!")

    session = klass(
        auto_refresh_url=auth_server.token_url,
        auto_refresh_kwargs={"client_id": client_id},
        token=token,
        token_updater=updater,
        client_id=client_id,
        scope=make_scope(scope),
        **kwargs,
    )

    return session
