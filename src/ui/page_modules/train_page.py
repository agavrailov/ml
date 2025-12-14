from __future__ import annotations

# Import UI components for modern styling
from src.ui import components


def _build_train_job_request_obj(
    *,
    train_freq: str,
    train_tsteps: int,
    train_lstm_units: int,
    train_lr: float,
    train_epochs: int,
    train_batch_size: int,
    train_n_layers: int,
    train_stateful: bool,
    train_features_to_use: list[str],
) -> dict:
    return {
        "frequency": train_freq,
        "tsteps": int(train_tsteps),
        "lstm_units": int(train_lstm_units),
        "learning_rate": float(train_lr),
        "epochs": int(train_epochs),
        "batch_size": int(train_batch_size),
        "n_lstm_layers": int(train_n_layers),
        "stateful": bool(train_stateful),
        "features_to_use": train_features_to_use,
    }


def _append_history(history: list[dict], row: dict, *, max_rows: int) -> list[dict]:
    out = list(history or [])
    out.append(row)
    if len(out) > int(max_rows):
        out = out[-int(max_rows) :]
    return out


def render_train_tab(
    *,
    st,
    pd,
    json,
    Path,
    os,
    FREQUENCY: str,
    TSTEPS: int,
    RESAMPLE_FREQUENCIES: list[str],
    TSTEPS_OPTIONS: list[int],
    LSTM_UNITS_OPTIONS: list[int],
    N_LSTM_LAYERS_OPTIONS: list[int],
    BATCH_SIZE_OPTIONS: list[int],
    STATEFUL_OPTIONS: list[bool],
    FEATURES_TO_USE_OPTIONS: list[list[str]],
    MODEL_REGISTRY_DIR: str,
    MAX_HISTORY_ROWS: int,
    get_run_hyperparameters,
    get_ui_state,
    load_json_history,
    save_json_history,
    format_timestamp,
    list_registry_models,
    promote_training_row,
) -> None:
    st.subheader("3. Train models (per frequency) & promote")

    ui_state = get_ui_state()
    train_state = ui_state.setdefault("training", {})
    if "history" not in train_state:
        train_state["history"] = load_json_history("training_history.json")

    # Prefill from last experiment if available, otherwise from defaults.
    prefill = train_state.get("train_prefill") or st.session_state.get("train_prefill")
    if prefill:
        default_freq = prefill["frequency"]
        default_tsteps = int(prefill["tsteps"])
    else:
        default_freq = FREQUENCY
        default_tsteps = TSTEPS

    # Keep one shared global frequency across tabs.
    _global_freq = st.session_state.get("global_frequency", default_freq)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = default_freq if default_freq in RESAMPLE_FREQUENCIES else RESAMPLE_FREQUENCIES[0]

    # Frequency is outside the form so changing it triggers a rerun (and thus
    # redraws the registry table below).
    train_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="train_freq_select_outside",
    )
    st.session_state["global_frequency"] = train_freq

    # Use a form to keep the parameter UI compact.
    with st.form("train_run_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
        with c1:
            try:
                train_tsteps_idx = TSTEPS_OPTIONS.index(default_tsteps)
            except ValueError:
                train_tsteps_idx = 0
            train_tsteps = st.selectbox(
                "TSTEPS",
                TSTEPS_OPTIONS,
                index=train_tsteps_idx,
                key="train_tsteps_select",
            )

        # Resolve effective hyperparameters for this run.
        base_hps = get_run_hyperparameters(frequency=train_freq, tsteps=train_tsteps)
        if prefill:
            base_hps.update(
                {
                    "lstm_units": int(prefill.get("lstm_units", base_hps["lstm_units"])),
                    "batch_size": int(prefill.get("batch_size", base_hps["batch_size"])),
                    "learning_rate": float(prefill.get("learning_rate", base_hps["learning_rate"])),
                    "epochs": int(prefill.get("epochs", base_hps["epochs"])),
                    "n_lstm_layers": int(prefill.get("n_lstm_layers", base_hps["n_lstm_layers"])),
                    "stateful": bool(prefill.get("stateful", base_hps["stateful"])),
                }
            )

        with c2:
            train_lstm_units = st.selectbox(
                "Units",
                LSTM_UNITS_OPTIONS,
                index=LSTM_UNITS_OPTIONS.index(base_hps["lstm_units"]) if base_hps["lstm_units"] in LSTM_UNITS_OPTIONS else 0,
                key="train_lstm_units_select",
            )

        with c3:
            train_n_layers = st.selectbox(
                "Layers",
                N_LSTM_LAYERS_OPTIONS,
                index=N_LSTM_LAYERS_OPTIONS.index(base_hps["n_lstm_layers"]) if base_hps["n_lstm_layers"] in N_LSTM_LAYERS_OPTIONS else 0,
                key="train_lstm_layers_select",
            )

        c4, c5, c6, c7 = st.columns([1.0, 1.0, 1.0, 1.4])
        with c4:
            train_batch_size = st.selectbox(
                "Batch",
                BATCH_SIZE_OPTIONS,
                index=BATCH_SIZE_OPTIONS.index(base_hps["batch_size"]) if base_hps["batch_size"] in BATCH_SIZE_OPTIONS else 0,
                key="train_batch_size_select",
            )

        with c5:
            lr_choices_t = [0.0005, 0.001, 0.003, 0.01]
            default_lr_t = float(base_hps["learning_rate"])
            if default_lr_t not in lr_choices_t:
                lr_choices_t = sorted(set(lr_choices_t + [default_lr_t]))
            train_lr = st.selectbox(
                "LR",
                lr_choices_t,
                index=lr_choices_t.index(default_lr_t) if default_lr_t in lr_choices_t else 0,
                key="train_lr_select",
            )

        with c6:
            train_epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=200,
                value=int(base_hps["epochs"]),
                step=1,
                key="train_epochs_number",
            )

        with c7:
            train_stateful = st.selectbox(
                "Stateful",
                STATEFUL_OPTIONS,
                index=STATEFUL_OPTIONS.index(base_hps["stateful"]) if base_hps["stateful"] in STATEFUL_OPTIONS else 0,
                key="train_stateful_select",
            )

        # Features are important but verbose; keep behind an expander.
        with st.expander("Advanced: feature set", expanded=False):
            if prefill and "features_to_use" in prefill:
                feats_prefill = prefill["features_to_use"].split(",")
                try:
                    train_feat_idx = next(
                        i for i, fs in enumerate(FEATURES_TO_USE_OPTIONS) if fs == feats_prefill
                    )
                except StopIteration:
                    train_feat_idx = 0
            else:
                train_feat_idx = 0

            train_feature_set_idx = st.selectbox(
                "Feature set",
                options=list(range(len(FEATURES_TO_USE_OPTIONS))),
                index=train_feat_idx,
                format_func=lambda i: f"Set {i + 1}: " + ", ".join(FEATURES_TO_USE_OPTIONS[i]),
                key="train_feature_set_select",
            )
            train_features_to_use = FEATURES_TO_USE_OPTIONS[train_feature_set_idx]

        submitted = st.form_submit_button("Train")

    # Async training job runner (out-of-process).
    # This keeps Streamlit reruns from interrupting long training runs.
    from src.core.contracts import TrainResult as _TrainResult
    from src.jobs import store as _job_store
    from src.jobs.types import JobType as _JobType
    import subprocess as _subprocess
    import sys as _sys
    import uuid as _uuid

    active_job_id = train_state.get("active_job_id")

    if active_job_id:
        job_status = _job_store.read_status(active_job_id)
        
        # Prepare result if job succeeded
        result = None
        if job_status and job_status.state == "SUCCEEDED":
            res_path = _job_store.result_path(active_job_id)
            if res_path.exists():
                try:
                    result = _job_store.read_json(res_path)
                except Exception:
                    pass
        
        # Professional job status panel with timeline and results
        components.render_job_status(
            st,
            job_id=active_job_id,
            job_type="train",
            status=job_status.to_dict() if job_status else {},
            result=result,
            on_refresh=lambda: st.rerun()
        )
        
        # Record run into UI history exactly once per job (after success)
        if job_status and job_status.state == "SUCCEEDED" and result:
            recorded = train_state.setdefault("recorded_job_ids", [])
            if active_job_id not in recorded:
                try:
                    req_obj = _job_store.read_json(_job_store.request_path(active_job_id))
                except Exception:
                    req_obj = {}
                
                row = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "frequency": req_obj.get("frequency", train_freq),
                    "tsteps": int(req_obj.get("tsteps", train_tsteps)),
                    "validation_loss": float(result.get("validation_loss", 0)),
                    "model_filename": result.get("model_filename", ""),
                    "bias_correction_filename": result.get("bias_correction_filename"),
                    "lstm_units": int(req_obj.get("lstm_units") or train_lstm_units),
                    "batch_size": int(req_obj.get("batch_size") or train_batch_size),
                    "learning_rate": float(req_obj.get("learning_rate") or train_lr),
                    "epochs": int(req_obj.get("epochs") or train_epochs),
                    "n_lstm_layers": int(req_obj.get("n_lstm_layers") or train_n_layers),
                    "stateful": bool(req_obj.get("stateful") if req_obj.get("stateful") is not None else train_stateful),
                    "optimizer_name": base_hps["optimizer_name"],
                    "loss_function": base_hps["loss_function"],
                    "features_to_use": req_obj.get("features_to_use") or train_features_to_use,
                }
                
                history: list[dict] = train_state.get("history", [])
                history = _append_history(history, row, max_rows=MAX_HISTORY_ROWS)
                train_state["history"] = history
                save_json_history("training_history.json", history)
                
                recorded.append(active_job_id)

    # Start a new job.
    if submitted:
        try:
            job_id = _uuid.uuid4().hex

            request_obj = _build_train_job_request_obj(
                train_freq=train_freq,
                train_tsteps=int(train_tsteps),
                train_lstm_units=int(train_lstm_units),
                train_lr=float(train_lr),
                train_epochs=int(train_epochs),
                train_batch_size=int(train_batch_size),
                train_n_layers=int(train_n_layers),
                train_stateful=bool(train_stateful),
                train_features_to_use=train_features_to_use,
            )

            _job_store.write_request(job_id, request_obj)

            # Capture stdout/stderr for debugging.
            log_path = _job_store.artifacts_dir(job_id) / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_f:
                _subprocess.Popen(
                    [
                        _sys.executable,
                        "-m",
                        "src.jobs.run",
                        "--job-id",
                        job_id,
                        "--job-type",
                        _JobType.TRAIN.value,
                        "--request",
                        str(_job_store.request_path(job_id)),
                    ],
                    stdout=log_f,
                    stderr=_subprocess.STDOUT,
                )

            train_state["active_job_id"] = job_id
            st.success(f"Started training job `{job_id}`. Use 'Refresh training job status' to monitor progress.")
        except Exception as exc:  # pragma: no cover
            st.error(f"Error starting training job: {exc}")

    # ---- Registry models table (filtered by frequency) ----
    st.markdown("### Models in registry (filtered by frequency)")

    best_hps_path_train = Path(__file__).resolve().parents[3] / "best_hyperparameters.json"

    registry_dir = Path(MODEL_REGISTRY_DIR)
    st.caption(f"Registry: `{registry_dir}`")

    # Build a lookup of validation_loss by model_filename from:
    # 1) UI training history (only includes runs started from the Streamlit UI)
    # 2) best_hyperparameters.json (best-only, but still helpful)
    val_loss_by_model: dict[str, float] = {}
    history_all = train_state.get("history", [])
    for r in history_all or []:
        try:
            name = r.get("model_filename")
            loss = r.get("validation_loss")
            if name and loss is not None:
                val_loss_by_model[str(name)] = float(loss)
        except Exception:
            continue

    if best_hps_path_train.exists():
        try:
            best_data = json.loads(best_hps_path_train.read_text(encoding="utf-8") or "{}")
            if isinstance(best_data, dict):
                for _freq, tsteps_data in best_data.items():
                    if not isinstance(tsteps_data, dict):
                        continue
                    for _t, metrics in tsteps_data.items():
                        if not isinstance(metrics, dict):
                            continue
                        name = metrics.get("model_filename")
                        if not name:
                            continue
                        try:
                            loss = float(metrics.get("validation_loss"))
                        except Exception:
                            continue
                        val_loss_by_model[str(name)] = loss
        except Exception:
            pass

    # List registry contents and filter by selected frequency.
    registry_rows = [r for r in list_registry_models(registry_dir) if r.get("frequency") == train_freq]

    if not registry_rows:
        st.info("No models found in the registry for this frequency.")
    else:
        for r in registry_rows:
            r["timestamp"] = format_timestamp(r.get("timestamp")) or r.get("timestamp")
            name = str(r.get("model_filename", ""))
            # Prefer per-model metrics sidecar, then fall back to UI history/best_hps lookup.
            if r.get("validation_loss") is None:
                r["validation_loss"] = val_loss_by_model.get(name)

        # Render a compact "table" with a per-row promote button.
        h_ts, h_loss, h_model, h_btn = st.columns([2.0, 1.0, 5.0, 0.9])
        h_ts.markdown("**timestamp**")
        h_loss.markdown("**val_loss**")
        h_model.markdown("**model**")
        h_btn.markdown("**promote**")

        for r in registry_rows:
            loss = r.get("validation_loss")
            loss_str = f"{float(loss):.6f}" if loss is not None else "(unknown)"
            ts_str = str(r.get("timestamp", ""))
            model_name = str(r.get("model_filename", ""))

            c_ts, c_loss, c_model, c_btn = st.columns([2.0, 1.0, 5.0, 0.9])
            c_ts.write(ts_str)
            c_loss.write(loss_str)
            c_model.write(f"`{model_name}`")

            if c_btn.button("↑", key=f"promote_{train_freq}_{model_name}"):
                promote_loss = loss
                if promote_loss is None:
                    st.warning(
                        "Selected model has no recorded val_loss; promoting with validation_loss=0.0."
                    )
                    promote_loss = 0.0

                promote_training_row(
                    row={
                        "validation_loss": float(promote_loss),
                        "model_filename": model_name,
                        "bias_correction_filename": r.get("bias_correction_filename"),
                    },
                    best_hps_path=best_hps_path_train,
                    frequency=str(r.get("frequency")),
                    tsteps=int(r.get("tsteps")),
                )
                st.success(f"Promoted: {model_name}")

    # Optional: show UI-recorded training runs for debugging/traceability.
    with st.expander("UI training history (optional)", expanded=False):
        filtered_ui = [
            r
            for r in (history_all or [])
            if isinstance(r, dict) and r.get("frequency") == train_freq
        ]
        if not filtered_ui:
            st.write("(none)")
        else:
            df_ui = pd.DataFrame(filtered_ui)
            if "timestamp" in df_ui.columns:
                df_ui["timestamp"] = df_ui["timestamp"].apply(format_timestamp)
            show_cols = [c for c in ["timestamp", "validation_loss", "tsteps", "model_filename"] if c in df_ui.columns]
            st.dataframe(df_ui[show_cols], width="stretch")

    # Clear prefill once the user has visited this tab.
    train_state.pop("train_prefill", None)
    st.session_state.pop("train_prefill", None)
