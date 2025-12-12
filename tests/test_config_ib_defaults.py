import pytest


def test_ib_config_defaults_to_ib_gateway_paper_port(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure defaults do not depend on the developer's environment.
    monkeypatch.delenv("TWS_HOST", raising=False)
    monkeypatch.delenv("TWS_PORT", raising=False)
    monkeypatch.delenv("TWS_CLIENT_ID", raising=False)

    from src.config import IbConfig

    cfg = IbConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 4002
    assert cfg.client_id == 1


def test_ib_config_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWS_HOST", "10.0.0.5")
    monkeypatch.setenv("TWS_PORT", "12345")
    monkeypatch.setenv("TWS_CLIENT_ID", "77")

    from src.config import IbConfig

    cfg = IbConfig()
    assert cfg.host == "10.0.0.5"
    assert cfg.port == 12345
    assert cfg.client_id == 77
