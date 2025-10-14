import io
import time
import webbrowser
from typing import List, Optional

import qrcode  # type: ignore
import requests
from requests_oauthlib import OAuth2Session  # type: ignore

from ..exceptions import AuthFlowError
from ..types import Plugin
from ..utils import ServerDetails, make_scope
from .utils import refresh_expired, scope_mismatch


def _make_qr(msg: str) -> str:
    qr = qrcode.QRCode()
    qr.add_data(msg)
    f = io.StringIO()
    qr.print_ascii(out=f)
    f.seek(0)
    return f.read()


def _prompt_user(partial_url: str, user_code: str, full_url: str) -> None:
    webbrowser.open(full_url)

    # Stole this prompt from AWS SSO
    print(
        f"""\
Attempting to automatically open the SSO authorization page in your default browser.
If the browser does not open or you wish to use a different device to authorize this
request, open the following URL:

{partial_url}

Then enter the code:

{user_code}"""
    )

    print(_make_qr(full_url))


def _poll_for_token(
    expires_in: float, interval: float, device_code: str, client_id: str, token_url: str
) -> dict:
    start = time.time()
    while (time.time() - start) < expires_in:
        # Sleep at start so we don't hit the server like, 30ms after we begin the process
        time.sleep(interval)

        res = requests.post(
            token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": client_id,
            },
        )

        if res.ok:
            break

        if res.status_code > 500:
            raise AuthFlowError("Server error talking to OIDC server")

        try:
            # Can fail if we didn't get back JSON
            data = res.json()
        except:
            res.raise_for_status()

        if data["error"] == "authorization_pending":
            continue
        elif data["error"] == "slow_down":
            # Should never get this, we're waiting correctly
            continue
        elif data["error"] == "expired_token":
            raise AuthFlowError("Device code timed out")
        elif data["error"] == "invalid_grant":
            raise AuthFlowError("Device code timed out or was invalid")
        elif data["error"] == "access_denied":
            raise AuthFlowError("Idiot")
        else:
            res.raise_for_status()
            continue

    res.raise_for_status()
    return res.json()


def device_code_flow(
    urls: ServerDetails, client_id: str, scope: List[str], aud: str
) -> dict:
    res = requests.post(
        urls.device_url,
        data={"client_id": client_id, "scope": make_scope(scope), "audience": aud},
    )
    res.raise_for_status()
    data = res.json()

    _prompt_user(
        data["verification_uri"], data["user_code"], data["verification_uri_complete"]
    )

    token = _poll_for_token(
        data["expires_in"],
        data["interval"],
        data["device_code"],
        client_id,
        urls.token_url,
    )

    if token["expires_in"]:
        token["expires_at"] = time.time() + token["expires_in"]

    return token


def make_device_code_session(
    oidc_url: str,
    client_id: str,
    audience: str,
    token: Optional[dict] = None,
    scope: Optional[List[str]] = None,
    *,
    klass=OAuth2Session,
    plugin: Optional[Plugin] = None,
    **kwargs,
):
    auth_server = ServerDetails.discover(oidc_url)
    scope = make_scope(scope)

    def updater(token: dict) -> None:
        if plugin:
            plugin.update(token)

    token = None if plugin is None else plugin.load()
    if token is None or refresh_expired(token, margin=15) or scope_mismatch(token, scope):
        token = device_code_flow(auth_server, client_id, scope, audience)
        updater(token)

    session = klass(
        auto_refresh_url=auth_server.token_url,
        auto_refresh_kwargs={"client_id": client_id},
        token=token,
        token_updater=updater,
        **kwargs,
    )

    return session
