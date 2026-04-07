from __future__ import annotations

from tests.isolation.singleton_boundary_helpers import (
    capture_exact_proxy_bootstrap_contract,
)


def test_no_proxy_omission_allowed() -> None:
    payload = capture_exact_proxy_bootstrap_contract()

    assert payload["omitted_proxies"] == []
    assert payload["forbidden_matches"] == []

    matrix = payload["matrix"]
    assert matrix["base.py"]["bound"] is True
    assert matrix["folder_paths_proxy.py"]["bound"] is True
    assert matrix["helper_proxies.py"]["bound"] is True
    assert matrix["model_management_proxy.py"]["bound"] is True
    assert matrix["progress_proxy.py"]["bound"] is True
    assert matrix["prompt_server_impl.py"]["bound"] is True
    assert matrix["utils_proxy.py"]["bound"] is True
    assert matrix["web_directory_proxy.py"]["bound"] is True
