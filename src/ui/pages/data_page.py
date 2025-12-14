from __future__ import annotations


def render_data_tab(
    *,
    st,
    pd,
    os,
    Path,
    project_root,
    RESAMPLE_FREQUENCIES: list[str],
    FREQUENCY: str,
    clean_raw_minute_data,
    convert_minute_to_timeframe,
    prepare_keras_input_data,
) -> None:
    st.subheader("1. Data ingestion & preparation")

    from src.config import RAW_DATA_CSV, PROCESSED_DATA_DIR, FREQUENCY as CFG_FREQ
    from src.data_quality import (
        analyze_raw_minute_data,
        compute_quality_kpi,
        format_quality_report,
        get_missing_trading_days,
    )

    st.markdown("### Raw minute data")
    st.write(f"Raw data CSV: `{RAW_DATA_CSV}`")

    st.markdown("#### Data quality checks")
    if st.button("Run data quality checks on raw minute data"):
        try:
            checks = analyze_raw_minute_data(RAW_DATA_CSV)
            if not checks:
                st.info("No checks were run (file may be missing or empty).")
            else:
                kpi = compute_quality_kpi(checks)

                # Overall KPI summary
                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                with c_kpi1:
                    st.metric("Data quality score", f"{kpi['score_0_100']:.1f} / 100")
                with c_kpi2:
                    st.metric("Checks passed", f"{kpi['n_pass']} / {kpi['n_total']}")
                with c_kpi3:
                    st.metric(
                        "Warnings / Failures",
                        f"{kpi['n_warn']} warn, {kpi['n_fail']} fail",
                    )

                # Detailed table of all checks
                checks_df = pd.DataFrame(checks).sort_values(["category", "id"])
                st.dataframe(checks_df, width="stretch")

                # Prominent status banner
                if kpi["n_fail"]:
                    st.error(
                        f"{kpi['n_fail']} checks FAILED. Review the table above before trusting the data.",
                    )
                elif kpi["n_warn"]:
                    st.warning(
                        f"All critical checks passed, but {kpi['n_warn']} checks have WARNINGS. "
                        "Review them for potential issues.",
                    )
                else:
                    st.success("All data quality checks passed.")

                # Downloadable plain-text/Markdown report
                report_text = format_quality_report(checks, kpi, dataset_name="Raw minute data")
                st.download_button(
                    label="Download data quality report",
                    data=report_text,
                    file_name="raw_data_quality_report.txt",
                    mime="text/plain",
                )
        except Exception as exc:  # pragma: no cover - UI convenience only
            st.error(f"Data quality checks failed: {exc}")

    st.markdown("#### Backfill missing trading days via TWS")
    if st.button("Backfill missing trading days from IB/TWS"):
        try:
            # Determine holiday-aware missing trading days based on NASDAQ calendar.
            missing_days = get_missing_trading_days(RAW_DATA_CSV, calendar_name="NASDAQ")
            if not missing_days:
                st.info("No missing trading days detected (holiday-aware NASDAQ calendar).")
            else:
                # Use the earliest and latest missing day as the backfill range.
                start_missing = min(missing_days)
                end_missing = max(missing_days)

                # IB's historical API uses an end timestamp that is effectively
                # exclusive; add one day so we cover the full last missing date.
                start_str = start_missing.strftime("%Y-%m-%d")
                end_str = (end_missing + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                import subprocess as _subprocess
                import sys as _sys

                cmd = [
                    _sys.executable,
                    "-m",
                    "src.data_ingestion",
                    "--start",
                    start_str,
                    "--end",
                    end_str,
                    "--strict-range",
                ]

                st.write(
                    "Triggering TWS backfill for missing trading days via CLI:",
                    " ".join(cmd),
                )
                with st.spinner(
                    f"Backfilling missing trading days from {start_str} to {end_str} via TWS...",
                ):
                    result = _subprocess.run(
                        cmd,
                        cwd=project_root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                if result.returncode == 0:
                    # Clean the raw minute data to sort and de-duplicate after backfill.
                    try:
                        clean_raw_minute_data(RAW_DATA_CSV)
                        st.success(
                            "Backfill completed and raw minute data re-cleaned. "
                            "Re-run data quality checks to verify the gaps are closed.",
                        )
                    except Exception as exc:  # pragma: no cover - best-effort cleanup
                        st.warning(
                            "Backfill completed, but cleaning the raw data failed: "
                            f"{exc}",
                        )
                else:
                    st.error(
                        f"Backfill CLI exited with code {result.returncode}. "
                        "See logs below.",
                    )

                if result.stdout:
                    with st.expander("Backfill log (stdout)"):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("Backfill log (stderr)"):
                        st.code(result.stderr)
        except Exception as exc:  # pragma: no cover - UI convenience only
            st.error(f"Backfill of missing trading days failed: {exc}")

    col_ingest1, col_ingest2 = st.columns(2)
    with col_ingest1:
        if st.button("Fetch/update raw NVDA data from TWS"):
            try:
                # Run the ingestion CLI in a subprocess to avoid importing ib_insync
                # (and its event-loop setup) into the Streamlit runtime.
                import subprocess
                import sys as _sys

                cmd = [_sys.executable, "-m", "src.data_ingestion"]

                status = st.empty()
                progress = st.progress(0.0)
                status.write(f"Running: {' '.join(cmd)} in {project_root}")

                with st.spinner(
                    "Fetching historical data via TWS... this may take several minutes."
                ):
                    result = subprocess.run(
                        cmd,
                        cwd=project_root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                progress.progress(1.0)

                if result.returncode == 0:
                    st.success("TWS historical ingestion completed successfully.")
                else:
                    st.error(f"Ingestion CLI exited with code {result.returncode}.")

                if result.stdout:
                    with st.expander("Ingestion log (stdout)"):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("Ingestion log (stderr)"):
                        st.code(result.stderr)
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to trigger ingestion: {exc}")

    with col_ingest2:
        if st.button("Clean raw minute data (sort, dedupe)"):
            try:
                clean_raw_minute_data(RAW_DATA_CSV)
                st.success("Raw minute data cleaned.")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to clean raw minute data: {exc}")

    st.markdown("### Hourly data & features")
    _global_freq = st.session_state.get("global_frequency", CFG_FREQ)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = RESAMPLE_FREQUENCIES[0]
    data_freq = st.selectbox(
        "Frequency for resampling & features",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="data_freq_select",
    )
    st.session_state["global_frequency"] = data_freq

    col_resample, col_features = st.columns(2)
    with col_resample:
        if st.button("Resample minute → hourly CSV"):
            try:
                os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
                convert_minute_to_timeframe(RAW_DATA_CSV, data_freq, PROCESSED_DATA_DIR)
                st.success(f"Resampled data saved for {data_freq}.")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to resample data: {exc}")

    with col_features:
        if st.button("Prepare features for Keras input"):
            try:
                from src.config import FEATURES_TO_USE_OPTIONS as _FEAT_OPTS
                from src.config import get_hourly_data_csv_path

                default_features = _FEAT_OPTS[0]
                hourly_path = get_hourly_data_csv_path(data_freq)
                df_feat, feat_cols = prepare_keras_input_data(hourly_path, default_features)
                st.session_state["data_feature_preview"] = (df_feat.head(200), feat_cols)
                st.success(f"Prepared features for {data_freq}: {feat_cols}")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to prepare features: {exc}")

    if "data_feature_preview" in st.session_state:
        df_prev, feat_cols_prev = st.session_state["data_feature_preview"]
        st.markdown("#### Feature preview (first 200 rows)")
        st.write(f"Columns: {feat_cols_prev}")
        st.dataframe(df_prev, width="stretch")

    _ = FREQUENCY  # currently unused, but kept to match the app-level signature
    _ = Path
