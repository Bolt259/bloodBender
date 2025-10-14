from dataclasses import dataclass
from typing import Type, TypeVar

import requests

TSelf = TypeVar("TSelf", bound="ServerDetails")


@dataclass
class ServerDetails:
    oidc_url: str
    auth_url: str
    token_url: str
    device_url: str

    @classmethod
    def discover(cls: Type[TSelf], oidc_url: str) -> TSelf:
        res = requests.get(oidc_url)
        res.raise_for_status()
        data = res.json()

        return cls(
            oidc_url=oidc_url,
            auth_url=data["authorization_endpoint"],
            token_url=data["token_endpoint"],
            device_url=data["device_authorization_endpoint"],
        )
