from types import SimpleNamespace

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util._helpers import get_auth_header  # noqa: E402


def node(auth_token: str = "", api_key: str = ""):
    return SimpleNamespace(
        hidden=SimpleNamespace(
            auth_token_comfy_org=auth_token,
            api_key_comfy_org=api_key,
        )
    )


def test_environment_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("API_KEY_COMFY_ORG", "environment-key")

    assert get_auth_header(node("account-token", "request-key")) == {
        "X-API-KEY": "environment-key"
    }


def test_hidden_auth_token_is_used_without_environment_key(monkeypatch):
    monkeypatch.delenv("API_KEY_COMFY_ORG", raising=False)

    assert get_auth_header(node("account-token", "request-key")) == {
        "Authorization": "Bearer account-token"
    }


def test_hidden_api_key_is_used_without_environment_key(monkeypatch):
    monkeypatch.delenv("API_KEY_COMFY_ORG", raising=False)

    assert get_auth_header(node(api_key="request-key")) == {
        "X-API-KEY": "request-key"
    }
