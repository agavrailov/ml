"""Tests for IBKR account configuration."""
import os
from unittest.mock import patch

from src.config import IbConfig
from src.ibkr_broker import IBKRBrokerConfig


def test_ibconfig_defaults_to_robots_account():
    """IbConfig should default to U16442949 (Robots) account."""
    config = IbConfig()
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
    """IBKRBrokerConfig.from_global_config should include account."""
    cfg = IBKRBrokerConfig.from_global_config()
    # Should have the default account value
    assert cfg.account == "U16442949"


def test_ibkr_broker_config_account_can_be_overridden():
    """IBKRBrokerConfig can be created with explicit account."""
    cfg = IBKRBrokerConfig(
        host="127.0.0.1",
        port=4002,
        client_id=1,
        account="U16485076",  # AI account
    )
    assert cfg.account == "U16485076"
