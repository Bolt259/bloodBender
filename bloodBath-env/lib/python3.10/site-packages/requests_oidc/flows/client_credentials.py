from typing import List, Optional

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session  # type: ignore

from ..types import Plugin
from ..utils import ServerDetails, make_scope


def make_client_credentials_session(
    oidc_url: str,
    client_id: str,
    client_secret: str,
    scope: Optional[List[str]] = None,
    *,
    klass=OAuth2Session,
    plugin: Optional[Plugin] = None,
    **kwargs,
) -> OAuth2Session:
    auth_server = ServerDetails.discover(oidc_url)
    client = BackendApplicationClient(client_id=client_id)
    scope = make_scope(scope)

    def updater(token: dict) -> None:
        if plugin:
            plugin.update(token)

    session = klass(
        client=client,
        auto_refresh_url=auth_server.token_url,
        token_updater=updater,
        scope=scope,
        **kwargs,
    )

    def refresh_token(url, *args, **kwargs) -> dict:
        return session.fetch_token(
            url, client_id=client_id, client_secret=client_secret
        )

    session.refresh_token = refresh_token
    token = session.refresh_token(session.auto_refresh_url)
    updater(token)

    return session
