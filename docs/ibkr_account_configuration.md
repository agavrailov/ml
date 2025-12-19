# IBKR Multi-Account Configuration

## Overview
The system supports multiple IBKR production accounts. You can configure which account to use via environment variable or code.

## Available Accounts
- **U16442949**: Robots (default)
- **U16452783**: Kids
- **U16485076**: AI
- **U16835894**: M7
- **U22210084**: ChinaTech

## Usage

### Method 1: Environment Variable (Recommended)
Set the `IBKR_ACCOUNT` environment variable before running the trading system:

**PowerShell:**
```powershell
$env:IBKR_ACCOUNT = "U16452783"
python -m src.ibkr_live_session --symbol NVDA --frequency 60min --backend IBKR_TWS
```

**Bash:**
```bash
export IBKR_ACCOUNT="U16452783"
python -m src.ibkr_live_session --symbol NVDA --frequency 60min --backend IBKR_TWS
```

**Windows CMD:**
```cmd
set IBKR_ACCOUNT=U16452783
python -m src.ibkr_live_session --symbol NVDA --frequency 60min --backend IBKR_TWS
```

### Method 2: Programmatic Configuration
When creating a broker instance in code:

```python
from src.ibkr_broker import IBKRBrokerConfig, IBKRBrokerTws

# Create config with specific account
config = IBKRBrokerConfig(
    host="127.0.0.1",
    port=4002,
    client_id=1,
    account="U16485076"  # AI account
)

# Create broker with config
broker = IBKRBrokerTws(config=config)
```

### Method 3: Using Global Config
The default configuration in `src/config.py` uses U16442949 (Robots) unless overridden by environment variable:

```python
from src.ibkr_broker import IBKRBrokerConfig

# Uses IB.account from config (defaults to U16442949)
config = IBKRBrokerConfig.from_global_config()
```

## Default Behavior
- If no `IBKR_ACCOUNT` environment variable is set, the system defaults to **U16442949** (Robots account)
- This ensures backward compatibility and safe defaults

## Notes
- The account ID must match one of your IBKR production accounts
- Ensure TWS or IB Gateway is configured to allow API connections for the selected account
- The account parameter is optional in `IBKRBrokerConfig`; if not provided, IBKR will use the default account configured in TWS/Gateway
