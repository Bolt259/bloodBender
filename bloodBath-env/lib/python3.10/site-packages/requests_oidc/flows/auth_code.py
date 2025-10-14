import webbrowser
from typing import List, Optional

from requests_oauthlib import OAuth2Session  # type: ignore

from ..types import Plugin
from ..utils import RedirectCatcher, ServerDetails, make_scope
from .utils import refresh_expired, scope_mismatch


def make_auth_code_session(
    oidc_url: str,
    client_id: str,
    port: int,
    scope: Optional[List[str]] = None,
    *,
    klass=OAuth2Session,
    plugin: Optional[Plugin] = None,
    **kwargs,
) -> OAuth2Session:
    # Docstring set below to leverage f-strings

    auth_server = ServerDetails.discover(oidc_url)
    redirect_catcher = RedirectCatcher(port)
    scope = make_scope(scope)

    def updater(token: dict) -> None:
        if plugin:
            plugin.update(token)

    token = None if plugin is None else plugin.load()
    if token is None or refresh_expired(token, margin=15) or scope_mismatch(token, scope):
        session = OAuth2Session(
            redirect_uri=redirect_catcher.redirect_uri,
            client_id=client_id,
            scope=scope,
        )
        auth_redirect_url, _ = session.authorization_url(auth_server.auth_url)
        webbrowser.open(auth_redirect_url)
        path = redirect_catcher.catch()
        token = session.fetch_token(auth_server.token_url, authorization_response=path)
        updater(token)

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


make_auth_code_session.__doc__ = f""" Create an `OAuth2Session <https://requests-oauthlib.readthedocs.io/en/latest/api.html#oauth-2-0-session>`_
    via web redirect, w/ automatic token management.

    After it is created, this session will behave as a regular
    `requests.Session <https://requests.readthedocs.io/en/latest/user/advanced/#session-objects>`_
    object,
    that injects the access token as an ``Authorization`` header. Do **not** use that session
    to call APIs that aren't the one you authenticated for, as that **will** leak your access
    token to third parties.

    To use this function, you'll need a **public** client w/ a ``redirect_uri`` set to
    ``{RedirectCatcher.FORMAT}``. Pick a unique ``port`` per client. Sharing the same one
    across different tools *may* work, but it's a bad assumption to rely on.

    The ``(client_id, port)`` tuple can be treated as constants within your code,
    and distributed as part of tooling that is built using this module.

    :param oidc_url: Path to an openid-connect server's `.well-known/openid-configuration`.
    :param client_id: Public client ID. This **must** be a public client w/o a secret.
    :param port: Port on localhost to redirect to from the auth server.
      ``{RedirectCatcher.FORMAT}`` must be a permitted ``redirect_uri`` for your client
      or the auth server will refuse to service your auth request.
    :param updater: Optional callback function to invoke whenever a token is fetched.
      This includes the first token fetch, and all refetches thereafter.


    .. _O2S: 
    .. _S: https://requests-oauthlib.readthedocs.io/en/latest/api.html#oauth-2-0-session
    """
