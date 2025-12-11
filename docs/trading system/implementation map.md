1. Backtester & strategy are the core  
◦  Make sure the strategy interface and backtester match trading_system_requirements + trading_system_strategy.
◦  Confirm: we can run backtests and get stable results.
2. Simulated/paper engine (no broker yet)  
◦  Implement a real‑time loop / service that:
▪  Consumes bars.
▪  Calls the same strategy.
▪  Uses a SimulatedBroker to simulate fills.
◦  Confirm: behavior matches backtests over the same period (within reason).
3. Broker abstraction  
◦  Introduce Broker interface, and adapt SimulatedBroker to it.
◦  Confirm: all existing flows work via Broker (no IBKR yet).
4. IBKR integration (paper only)  
◦  Implement IBKRBroker behind Broker.
◦  Implement RiskManagedBroker.
◦  Wire it via the BROKER_BACKEND config (SIM vs IBKR_PAPER).
◦  Confirm:  
▪  Small test trades on paper account appear in TWS.  
▪  Logs and internal PnL make sense.