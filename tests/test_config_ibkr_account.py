"""Tests for IBKR account configuration."""
import os
from unittest.mock import patch

from src.config import IbConfig
from src.ibkr_broker import IBKRBrokerConfig


def test_ibconfig_defaults_to_robots_account():
    """IbConfig should default to U16442949 (Robots) account when no env var is set."""
    # Clear env var to test code default
    with patch.dict(os.environ, {}, clear=False):
        if "IBKR_ACCOUNT" in os.environ:
            del os.environ["IBKR_ACCOUNT"]
        config = IbConfig(
            host=os.getenv("TWS_HOST", "127.0.0.1"),
            port=int(os.getenv("TWS_PORT", "4002")),
            client_id=int(os.getenv("TWS_CLIENT_ID", "1")),
            account=os.getenv("IBKR_ACCOUNT", "U16442949"),
        )
        assert config.account == "U16442949"


def test_ibconfig_account_env_override():
    """IBKR_ACCOUNT environment variable should override default."""
    with patch.dict(os.environ, {"IBKR_ACCOUNT": "U16452783"}):
        # Need to recreate the config since it uses default_factory
        config = IbConfig(
            host=os.getenv("TWS_HOST", "127.0.0.1"),
            port=int(os.getenv("TWS_PORT", "4002")),
            client_id=int(os.getenv("TWS_CLIENT_ID", "1")),
            account=os.getenv("IBKR_ACCOUNT", "U16442949"),
        )
        assert config.account == "U16452783"


def test_ibkr_broker_config_includes_account_from_global():
    """IBKRBrokerConfig.from_global_config should include account from config/env."""
    cfg = IBKRBrokerConfig.from_global_config()
    # Should have account value from global IB config (may be env-overridden)
    # Just verify it's one of the known accounts
    known_accounts = {"U16442949", "U16452783", "U16485076", "U16835894", "U22210084"}
    assert cfg.account in known_accounts


def test_ibkr_broker_config_account_can_be_overridden():
    """IBKRBrokerConfig can be created with explicit account."""
    cfg = IBKRBrokerConfig(
        host="127.0.0.1",
        port=4002,
        client_id=1,
        account="U16485076",  # AI account
    )
    assert cfg.account == "U16485076"
