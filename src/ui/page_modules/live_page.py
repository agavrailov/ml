"""Live ops dashboard page.

Reads live trading events from JSONL logs and presents an observability dashboard.
Event schemas are defined in src.live.contracts.
"""

from __future__ import annotations


def render_live_tab(
    *,
    st,
    pd,
    plt,
) -> None:
    st.subheader("0. Live Ops Dashboard")

    from datetime import datetime, timezone

    try:  # optional dependency
        from streamlit_autorefresh import st_autorefresh

        _HAVE_AUTOREFRESH = True
    except Exception:  # pragma: no cover
        st_autorefresh = None  # type: ignore[assignment]
        _HAVE_AUTOREFRESH = False

    from src.live_log import (
        append_event,
        default_live_dir,
        is_kill_switch_enabled,
        kill_switch_path,
        list_logs,
        read_events,
        set_kill_switch,
    )

    live_dir = default_live_dir()

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.caption(f"Live log directory: `{live_dir}`")
    with col_b:
        st.caption(f"Kill switch file: `{kill_switch_path()}`")
    with col_c:
        optimization_running = bool(st.session_state.get("optimization_running", False))
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=(not optimization_running),
            disabled=optimization_running,
        )
        if optimization_running:
            st.caption("Auto-refresh is temporarily disabled while optimization is running.")
        elif auto_refresh and _HAVE_AUTOREFRESH:
            # Streamlit UI updates on reruns; this triggers a rerun every 30 seconds.
            st_autorefresh(interval=30_000, key="live_autorefresh")
        elif auto_refresh and not _HAVE_AUTOREFRESH:
            st.caption("Auto-refresh unavailable (missing streamlit-autorefresh).")

        # Manual fallback.
        st.button("Refresh now")

    # Kill switch controls.
    ks_enabled = bool(is_kill_switch_enabled())
    cks1, cks2, cks3 = st.columns([1, 1, 2])
    with cks1:
        if st.button("Enable kill switch", disabled=ks_enabled):
            set_kill_switch(enabled=True)
            st.success("Kill switch enabled: new order submissions should be blocked.")
    with cks2:
        if st.button("Disable kill switch", disabled=not ks_enabled):
            set_kill_switch(enabled=False)
            st.success("Kill switch disabled.")
    with cks3:
        st.write(f"**Kill switch**: {'ENABLED' if ks_enabled else 'disabled'}")

    logs = list_logs()
    if not logs:
        st.info(
            "No live session logs found yet. Start a session via the CLI (e.g. "
            "`python -m src.ibkr_live_session --symbol NVDA --frequency 60min`) "
            "to generate a JSONL log under ui_state/live/."
        )
        return

    # Session selector.
    options = [
        f"{info.run_id} | {info.symbol} {info.frequency} | {info.path.name}" for info in logs
    ]
    selected = st.selectbox("Select session", options, index=0)
    selected_idx = options.index(selected)
    log_info = logs[selected_idx]
    log_path = log_info.path

    # Session annotations (Step 3b): allow the operator to append notes to the log.
    st.markdown("### Session notes")
    note_key = f"live_note_{log_path.name}"
    note_text = st.text_input(
        "Add a note to this session (stored in the JSONL log)",
        value="",
        key=note_key,
    )
    col_note1, col_note2 = st.columns([1, 3])
    with col_note1:
        if st.button("Append note", key=f"append_note_btn_{log_path.name}"):
            text = (note_text or "").strip()
            if not text:
                st.warning("Note text is empty.")
            else:
                try:
                    append_event(
                        log_path,
                        {
                            "type": "note",
                            "run_id": log_info.run_id,
                            "symbol": log_info.symbol,
                            "frequency": log_info.frequency,
                            "source": "ui",
                            "message": text,
                        },
                    )
                    st.success("Note appended.")
                    # Clear field by resetting session_state.
                    st.session_state[note_key] = ""
                except Exception as exc:  # pragma: no cover - UI convenience
                    st.error(f"Failed to append note: {exc}")
    with col_note2:
        st.caption("Use notes for: why you paused, parameter changes, incident context, etc.")

    # Load recent events (bounded for UI responsiveness).
    events = read_events(log_path, max_events=2000)

    def _latest(event_type: str):
        for e in reversed(events):
            if e.get("type") == event_type:
                return e
        return None

    last_run_start = _latest("run_start")
    last_run_end = _latest("run_end")
    last_bar = _latest("bar")
    last_decision = _latest("decision")
    last_order = _latest("order_submitted")
    last_broker_connected = _latest("broker_connected")
    last_broker_snapshot = _latest("broker_snapshot")
    last_error = _latest("error")

    now_utc = datetime.now(timezone.utc)

    def _parse_frequency_minutes(freq: str | None) -> int | None:
        if not freq:
            return None
        f = str(freq).strip().lower()
        try:
            if f.endswith("min"):
                return int(f.replace("min", "").strip())
            if f in {"1h", "1hr", "1hour"}:
                return 60
            if f in {"4h", "4hr", "4hour"}:
                return 240
        except Exception:
            return None
        return None

    def _parse_dt(val: object) -> datetime | None:
        if isinstance(val, datetime):
            t = val
            return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)
        if isinstance(val, str):
            try:
                t = datetime.fromisoformat(val)
                return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)
            except Exception:
                return None
        return None

    # Derive high-level engine status from run_start/run_end.
    status = "UNKNOWN"
    if last_run_start is not None and last_run_end is None:
        status = "RUNNING"
    if last_run_end is not None:
        status = "STOPPED"

    # Data freshness + STALE logic.
    last_bar_time = _parse_dt(last_bar.get("bar_time")) if last_bar else None
    bar_age_min: float | None = None
    if last_bar_time is not None:
        bar_age_min = (now_utc - last_bar_time).total_seconds() / 60.0

    freq_min = None
    if last_run_start is not None:
        freq_min = _parse_frequency_minutes(last_run_start.get("frequency"))
    if freq_min is None:
        freq_min = _parse_frequency_minutes(log_info.frequency)

    # If we know the bar cadence, consider STALE when we're far beyond expected.
    # Conservative defaults: at least 5 minutes, otherwise 2x bar period.
    stale_threshold_min = 5.0
    if isinstance(freq_min, int) and freq_min > 0:
        stale_threshold_min = float(max(5, 2 * freq_min))

    data_feed_status = "N/A"
    data_freshness_label = "N/A"
    is_stale = False
    if status == "RUNNING":
        if bar_age_min is None:
            data_feed_status = "NO_BARS_YET"
        else:
            is_stale = bar_age_min > stale_threshold_min
            data_feed_status = "STALE" if is_stale else "OK"
            data_freshness_label = f"{bar_age_min:.1f} min ago"
    else:
        if bar_age_min is not None:
            data_freshness_label = f"{bar_age_min:.1f} min ago"
        data_feed_status = "STOPPED"

    # Snapshot freshness.
    snap_time = _parse_dt(last_broker_snapshot.get("ts_utc")) if last_broker_snapshot else None
    snap_age_min: float | None = None
    if snap_time is not None:
        snap_age_min = (now_utc - snap_time).total_seconds() / 60.0

    # Prominent status banner.
    if last_error is not None:
        st.error("Last event is an ERROR. See the Errors section below.")
    elif bool(is_kill_switch_enabled()):
        st.warning("Kill switch is ENABLED: new order submissions should be blocked.")
    elif status == "RUNNING" and is_stale:
        st.warning(
            f"Live session appears STALE: last bar was {data_freshness_label} (threshold {stale_threshold_min:.0f} min)."
        )
    elif status == "RUNNING":
        st.success("Live session running.")
    elif status == "STOPPED":
        st.info("Session is stopped (log shows run_end).")

    # KPI cards (Tier 0-ish).
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Engine", status)
    with k2:
        st.metric(
            "Data feed",
            data_feed_status,
            delta=data_freshness_label if data_freshness_label != "N/A" else None,
        )
    with k3:
        broker_status = "UNKNOWN"
        if last_broker_connected is not None and status == "RUNNING":
            broker_status = "CONNECTED"
        elif status == "STOPPED":
            broker_status = "STOPPED"
        st.metric("Broker", broker_status)
    with k4:
        # Prefer broker snapshot when present.
        pos_state = "UNKNOWN"
        if last_broker_snapshot is not None:
            try:
                positions = last_broker_snapshot.get("positions") or []
                # If any non-zero position exists, consider it OPEN.
                nonzero = False
                for p in positions:
                    try:
                        qty = float(p.get("quantity", 0.0))
                        if abs(qty) > 1e-9:
                            nonzero = True
                            break
                    except Exception:
                        continue
                pos_state = "OPEN" if nonzero else "FLAT"
            except Exception:
                pos_state = "UNKNOWN"
        else:
            pos_state = "OPEN" if last_order is not None else "FLAT"
        st.metric("Position", pos_state)
    with k5:
        if last_decision is not None:
            st.metric("Last decision", str(last_decision.get("action", "")))
        else:
            st.metric("Last decision", "(none)")
    with k6:
        if last_order is not None:
            st.metric("Last order_id", str(last_order.get("order_id", "")))
        else:
            st.metric("Last order_id", "(none)")

    # Show recent notes (most recent first).
    notes = [e for e in events if e.get("type") == "note"]
    if notes:
        with st.expander("Recent notes", expanded=False):
            df_notes = pd.DataFrame(notes[-20:])
            keep_cols = [c for c in ["ts_utc", "source", "message"] if c in df_notes.columns]
            st.dataframe(df_notes[keep_cols].iloc[::-1], width="stretch")

    # Broker snapshot (Step 2): show "broker truth" derived from latest snapshot event.
    st.markdown("### Broker snapshot")
    if snap_age_min is not None:
        st.caption(f"Snapshot freshness: {snap_age_min:.1f} min ago")
    if last_broker_snapshot is None:
        st.info("No broker snapshots logged yet for this session.")
    else:
        try:
            positions = last_broker_snapshot.get("positions") or []
            open_orders = last_broker_snapshot.get("open_orders") or []
            acct = last_broker_snapshot.get("account_summary") or {}

            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Open orders", str(len(open_orders)))
            with s2:
                st.metric("Positions", str(len(positions)))
            with s3:
                # Prefer a common equity tag when present; otherwise show number of keys.
                eq = None
                key = ""
                for key in ("NetLiquidation", "EquityWithLoanValue", "TotalCashValue"):
                    if key in acct:
                        eq = acct.get(key)
                        break
                st.metric(
                    "Account summary",
                    f"{key}={eq}" if eq is not None else f"{len(acct)} fields",
                )

            with st.expander("Positions (latest snapshot)"):
                if positions:
                    st.dataframe(pd.DataFrame(positions), width="stretch")
                else:
                    st.write("(none)")

            with st.expander("Open orders (latest snapshot)"):
                if open_orders:
                    st.dataframe(pd.DataFrame(open_orders), width="stretch")
                else:
                    st.write("(none)")

            with st.expander("Account summary (latest snapshot)"):
                if acct:
                    st.dataframe(pd.DataFrame([acct]), width="stretch")
                else:
                    st.write("(empty)")

        except Exception as exc:  # pragma: no cover - UI convenience
            st.error(f"Failed to render broker snapshot: {exc}")

    # Recent decisions & orders.
    st.markdown("### Recent activity")
    col_d, col_o = st.columns(2)

    with col_d:
        st.markdown("#### Decisions")
        dec = [e for e in events if e.get("type") == "decision"]
        if dec:
            df = pd.DataFrame(dec[-50:])
            keep_cols = [
                c
                for c in [
                    "ts_utc",
                    "bar_time",
                    "action",
                    "direction",
                    "size",
                    "close",
                    "predicted_price",
                    "blocked_by_kill_switch",
                ]
                if c in df.columns
            ]
            st.dataframe(df[keep_cols].iloc[::-1], width="stretch")
        else:
            st.info("No decision events yet.")

    with col_o:
        st.markdown("#### Orders")
        ords = [e for e in events if e.get("type") == "order_submitted"]
        if ords:
            df = pd.DataFrame(ords[-50:])
            keep_cols = [
                c
                for c in ["ts_utc", "bar_time", "order_id", "direction", "size"]
                if c in df.columns
            ]
            st.dataframe(df[keep_cols].iloc[::-1], width="stretch")
        else:
            st.info("No order submissions yet.")

    # Charts (Step 3a): price + prediction + trade markers from the event log.
    st.markdown("### Charts")
    with st.expander("Price / prediction / trades", expanded=True):
        bars = [e for e in events if e.get("type") == "bar"]
        decisions = [e for e in events if e.get("type") == "decision"]
        orders = [e for e in events if e.get("type") == "order_submitted"]

        if not bars:
            st.info("No bar events yet.")
        else:
            df_bar = pd.DataFrame(bars)
            # Normalize bar_time.
            if "bar_time" in df_bar.columns:
                try:
                    df_bar["bar_time"] = pd.to_datetime(
                        df_bar["bar_time"], utc=True, errors="coerce"
                    )
                except Exception:
                    pass

            cols = [c for c in ["bar_time", "close"] if c in df_bar.columns]
            df_bar = df_bar[cols].copy() if cols else df_bar.copy()
            if "bar_time" in df_bar.columns:
                df_bar = (
                    df_bar.dropna(subset=["bar_time"])
                    .drop_duplicates(subset=["bar_time"], keep="last")
                    .sort_values("bar_time")
                )

            # Predicted price from decisions (if present).
            df_pred = None
            if decisions:
                df_d = pd.DataFrame(decisions)
                if "bar_time" in df_d.columns:
                    try:
                        df_d["bar_time"] = pd.to_datetime(
                            df_d["bar_time"], utc=True, errors="coerce"
                        )
                    except Exception:
                        pass
                if "predicted_price" in df_d.columns and "bar_time" in df_d.columns:
                    df_pred = (
                        df_d[["bar_time", "predicted_price"]]
                        .dropna(subset=["bar_time"])
                        .drop_duplicates(subset=["bar_time"], keep="last")
                        .sort_values("bar_time")
                    )

            plot_df = df_bar
            if df_pred is not None and "bar_time" in plot_df.columns:
                plot_df = plot_df.merge(df_pred, on="bar_time", how="left")

            if plot_df.empty:
                st.info("Not enough data to plot.")
            else:
                # Use matplotlib for consistent style with the rest of the app.
                fig, ax = plt.subplots(figsize=(12, 4))
                x = plot_df["bar_time"] if "bar_time" in plot_df.columns else plot_df.index

                if "close" in plot_df.columns:
                    ax.plot(x, plot_df["close"], label="Close", color="tab:blue")

                if "predicted_price" in plot_df.columns:
                    ax.plot(
                        x,
                        plot_df["predicted_price"],
                        label="Predicted",
                        color="tab:orange",
                        alpha=0.8,
                    )

                # Mark submitted orders (best-effort).
                if orders:
                    df_o = pd.DataFrame(orders)
                    if "bar_time" in df_o.columns:
                        try:
                            df_o["bar_time"] = pd.to_datetime(
                                df_o["bar_time"], utc=True, errors="coerce"
                            )
                        except Exception:
                            pass
                    if (
                        "bar_time" in df_o.columns
                        and "close" in plot_df.columns
                        and "bar_time" in plot_df.columns
                    ):
                        # Align order times to plotted closes.
                        tmp = plot_df[["bar_time", "close"]].merge(
                            df_o[["bar_time", "direction"]].dropna(subset=["bar_time"]),
                            on="bar_time",
                            how="inner",
                        )
                        if not tmp.empty:
                            # Direction +1/-1 -> marker shape.
                            buys = tmp[tmp.get("direction", 0) >= 0]
                            sells = tmp[tmp.get("direction", 0) < 0]
                            if not buys.empty:
                                ax.scatter(
                                    buys["bar_time"],
                                    buys["close"],
                                    marker="^",
                                    color="green",
                                    s=40,
                                    label="Order (BUY)",
                                )
                            if not sells.empty:
                                ax.scatter(
                                    sells["bar_time"],
                                    sells["close"],
                                    marker="v",
                                    color="red",
                                    s=40,
                                    label="Order (SELL)",
                                )

                ax.set_title("Live: Close vs Predicted with order markers")
                ax.set_xlabel("Time")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()
                st.pyplot(fig)

    # Audit log (Tier 2): filter/search/export over the event timeline.
    st.markdown("### Audit log")
    import json

    all_types = sorted({str(e.get("type")) for e in events if e.get("type") is not None})
    default_types = [
        t
        for t in [
            "decision",
            "order_submitted",
            "order_status",
            "fill",
            "order_rejected",
            "error",
            "note",
        ]
        if t in all_types
    ]

    col_f1, col_f2, col_f3 = st.columns([2, 1, 2])
    with col_f1:
        selected_types = st.multiselect(
            "Event types", options=all_types, default=default_types
        )
    with col_f2:
        max_rows = int(
            st.number_input("Max rows", min_value=50, max_value=5000, value=500, step=50)
        )
    with col_f3:
        order_id_query = st.text_input("Search order_id contains", value="")

    filtered = [
        e for e in events if not selected_types or e.get("type") in set(selected_types)
    ]
    if order_id_query.strip():
        q = order_id_query.strip().lower()
        filtered = [
            e for e in filtered if q in str(e.get("order_id", "")).lower()
        ]

    # Keep the most recent items (timeline is stored oldest->newest).
    filtered = filtered[-max_rows:]

    st.caption(f"Filtered events: {len(filtered)} / {len(events)}")

    if filtered:
        df_a = pd.DataFrame(filtered)
        keep_cols = [
            c
            for c in [
                "ts_utc",
                "type",
                "bar_time",
                "where",
                "order_id",
                "status",
                "fill_quantity",
                "filled_quantity_total",
                "avg_fill_price",
                "action",
                "direction",
                "size",
                "blocked_by_kill_switch",
                "error",
                "message",
            ]
            if c in df_a.columns
        ]
        if keep_cols:
            st.dataframe(df_a[keep_cols].iloc[::-1], width="stretch")
        else:
            st.dataframe(df_a.iloc[::-1], width="stretch")

        # Export filtered view.
        jsonl_text = "\n".join(json.dumps(e, ensure_ascii=False) for e in filtered) + "\n"
        st.download_button(
            "Download filtered JSONL",
            data=jsonl_text,
            file_name=f"filtered_{log_path.name}",
            mime="application/jsonl",
        )
        try:
            st.download_button(
                "Download filtered CSV",
                data=df_a.to_csv(index=False),
                file_name=f"filtered_{log_path.stem}.csv",
                mime="text/csv",
            )
        except Exception:
            # CSV export can fail if complex/nested objects are present.
            pass
    else:
        st.info("No events match the current filters.")

    # Errors (Tier 2 drill-down).
    if last_error is not None:
        st.markdown("### Errors")
        err_events = [e for e in events if e.get("type") == "error"]
        for e in err_events[-5:][::-1]:
            where = e.get("where", "")
            err = e.get("error", "")
            ts = e.get("ts_utc", "")
            with st.expander(f"{ts} | {where} | {err}"):
                st.code(str(e.get("traceback", "")))

    # Export/download.
    st.markdown("### Export")
    try:
        raw_text = log_path.read_text(encoding="utf-8")
        st.download_button(
            "Download raw JSONL log",
            data=raw_text,
            file_name=log_path.name,
            mime="application/jsonl",
        )
    except Exception as exc:  # pragma: no cover - UI convenience
        st.error(f"Failed to read log file for download: {exc}")
