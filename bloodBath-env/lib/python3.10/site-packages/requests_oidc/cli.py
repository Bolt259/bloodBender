import typer
from typing import Annotated, Never
from requests_oauthlib import OAuth2Session
import time
import requests_oidc as flows
from pathlib import Path
from .plugins import PathPlugin

app = typer.Typer(no_args_is_help=True)
daemon_app = typer.Typer(no_args_is_help=True)
app.add_typer(daemon_app, name='daemon')


OIDC_URL = typer.Argument(
    envvar="OIDC_URL",
    help="URL of OIDC `.well-known/openid-configuration` JSON object.",
)
OIDC_CLIENT_ID = typer.Argument(
    envvar="OIDC_CLIENT_ID",
    help="OIDC Client ID, should be unique per-program.",
)
OIDC_CLIENT_SECRET = typer.Argument(
    envvar="OIDC_CLIENT_SECRET",
    help="OIDC Client Secret. Keep it Safe.",
)
OIDC_CALLBACK_PORT = typer.Argument(
    envvar="OIDC_CALLBACK_PORT",
    help="Localhost port used for browser-driven auth flows.",
)
OIDC_AUDIENCE = typer.Argument(
    envvar="OIDC_AUDIENCE",
    help="Service(s) you wish to be authenticated to talk to.",
)
OIDC_OFFLINE = typer.Option(
    envvar="OIDC_OFFLINE",
    help="""\
Request an offline access token. So long as there refreshed occasionally, these won't \
expire.""",
)
OIDC_MARGIN = typer.Option(
    envvar="OIDC_MARGIN",
    help="How many seconds before access token expirey to wait to refresh.",
)

def hibernate(session: OAuth2Session, margin: int) -> Never:
    while True:
        time.sleep(session.token["expires_in"] - margin)
        # requests_oauthlib is very bad, why isn't this kicked off from .refresh_token?
        token = session.refresh_token(session.auto_refresh_url)
        session.token_updater(token)

@daemon_app.command()
def auth_code(
    path: Path,
    oidc_url: Annotated[str, OIDC_URL],
    client_id: Annotated[str, OIDC_CLIENT_ID],
    callback_port: Annotated[int, OIDC_CALLBACK_PORT],
    offline_access: Annotated[bool, OIDC_OFFLINE] = False,
    margin: Annotated[int, OIDC_MARGIN] = 15,
) -> None:
    scope = []

    if offline_access:
        scope.append("offline_access")

    session = flows.make_auth_code_session(
        oidc_url=oidc_url,
        client_id=client_id,
        port=callback_port,
        scope=scope,
        plugin=PathPlugin(path),
    )

    hibernate(session, margin)


@daemon_app.command()
def device_code(
    path: Path,
    oidc_url: Annotated[str, OIDC_URL],
    client_id: Annotated[str, OIDC_CLIENT_ID],
    audience: Annotated[str, OIDC_AUDIENCE],
    offline_access: Annotated[bool, OIDC_OFFLINE] = False,
    margin: Annotated[int, OIDC_MARGIN] = 15,
) -> None:
    scope = []

    if offline_access:
        scope.append("offline_access")

    session = flows.make_device_code_session(
        path=path,
        oidc_url=oidc_url,
        client_id=client_id,
        audience=audience,
        scope=scope,
        plugin=PathPlugin(path),
    )

    hibernate(session, margin)


@daemon_app.command()
def client_credentials(
    path: Path,
    oidc_url: Annotated[str, OIDC_URL],
    client_id: Annotated[str, OIDC_CLIENT_ID],
    client_secret: Annotated[str, OIDC_CLIENT_SECRET],
    margin: Annotated[int, OIDC_MARGIN] = 15,
) -> None:
    session = flows.make_client_credentials_session(
        oidc_url=oidc_url,
        client_id=client_id,
        client_secret=client_secret,
        plugin=PathPlugin(path),
    )

    hibernate(session, margin)
