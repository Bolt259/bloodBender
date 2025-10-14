import os

from .flows.auth_code import make_auth_code_session
from .flows.client_credentials import make_client_credentials_session
from .flows.device_code import make_device_code_session
from .flows.token import make_token_session

# BS to work around oauthlib's wonkyness, they don't make this configurable w/o envvars
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "True")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "True")
