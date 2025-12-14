from __future__ import annotations

# Import UI components for modern styling
from src.ui import components


def render_experiments_tab(
    *,
    st,
    pd,
    os,
    train_model,
    FREQUENCY: str,
    TSTEPS: int,
    RESAMPLE_FREQUENCIES: list[str],
    TSTEPS_OPTIONS: list[int],
    LSTM_UNITS_OPTIONS: list[int],
    N_LSTM_LAYERS_OPTIONS: list[int],
    BATCH_SIZE_OPTIONS: list[int],
    STATEFUL_OPTIONS: list[bool],
    FEATURES_TO_USE_OPTIONS: list[list[str]],
    MAX_HISTORY_ROWS: int,
    get_run_hyperparameters,
    get_ui_state,
    load_json_history,
    save_json_history,
) -> None:
    st.subheader("2. Hyperparameter experiments (no promotion)")

    ui_state = get_ui_state()
    exp_state = ui_state.setdefault("experiments", {})
    if "runs" not in exp_state:
        exp_state["runs"] = load_json_history("experiments_runs.json")

    # Frequency / TSTEPS for experiments.
    _global_freq = st.session_state.get("global_frequency", FREQUENCY)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = RESAMPLE_FREQUENCIES[0]
    exp_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="exp_freq_select",
    )
    st.session_state["global_frequency"] = exp_freq
    try:
        exp_tsteps_idx = TSTEPS_OPTIONS.index(TSTEPS)
    except ValueError:
        exp_tsteps_idx = 0
    exp_tsteps = st.selectbox(
        "Sequence length (TSTEPS)",
        TSTEPS_OPTIONS,
        index=exp_tsteps_idx,
        key="exp_tsteps_select",
    )

    # Use current best hyperparameters as defaults for experiments.
    exp_hps = get_run_hyperparameters(frequency=exp_freq, tsteps=exp_tsteps)

    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            idx_units = LSTM_UNITS_OPTIONS.index(exp_hps["lstm_units"])
        except ValueError:
            idx_units = 0
        exp_lstm_units = st.selectbox(
            "LSTM units",
            LSTM_UNITS_OPTIONS,
            index=idx_units,
            key="exp_lstm_units_select",
        )

        try:
            idx_layers = N_LSTM_LAYERS_OPTIONS.index(exp_hps["n_lstm_layers"])
        except ValueError:
            idx_layers = 0
        exp_n_layers = st.selectbox(
            "LSTM layers",
            N_LSTM_LAYERS_OPTIONS,
            index=idx_layers,
            key="exp_lstm_layers_select",
        )

    with c2:
        try:
            idx_bs = BATCH_SIZE_OPTIONS.index(exp_hps["batch_size"])
        except ValueError:
            idx_bs = 0
        exp_batch_size = st.selectbox(
            "Batch size",
            BATCH_SIZE_OPTIONS,
            index=idx_bs,
            key="exp_batch_size_select",
        )

        exp_epochs = st.slider(
            "Epochs (short experiments)",
            min_value=1,
            max_value=50,
            value=min(10, int(exp_hps["epochs"])),
            key="exp_epochs_slider",
        )

    with c3:
        lr_choices = [0.0005, 0.001, 0.003, 0.01]
        default_lr = float(exp_hps["learning_rate"])
        if default_lr not in lr_choices:
            lr_choices = sorted(set(lr_choices + [default_lr]))
        try:
            idx_lr = lr_choices.index(default_lr)
        except ValueError:
            idx_lr = 0
        exp_lr = st.selectbox(
            "Learning rate",
            lr_choices,
            index=idx_lr,
            key="exp_lr_select",
        )

        try:
            idx_stateful = STATEFUL_OPTIONS.index(exp_hps["stateful"])
        except ValueError:
            idx_stateful = 0
        exp_stateful = st.selectbox(
            "Stateful LSTM",
            STATEFUL_OPTIONS,
            index=idx_stateful,
            key="exp_stateful_select",
        )

    exp_feature_set_idx = st.selectbox(
        "Feature set",
        options=list(range(len(FEATURES_TO_USE_OPTIONS))),
        format_func=lambda i: f"Set {i + 1}: " + ", ".join(FEATURES_TO_USE_OPTIONS[i]),
        key="exp_feature_set_select",
    )
    exp_features_to_use = FEATURES_TO_USE_OPTIONS[exp_feature_set_idx]

    st.caption("Experiments run shorter trainings and DO NOT update best_hyperparameters.json.")

    if st.button("Run single experiment"):
        progress = st.progress(0.0)
        status = st.empty()
        try:
            status.write("Running short training experiment...")
            progress.progress(0.1)

            result = train_model(
                frequency=exp_freq,
                tsteps=exp_tsteps,
                lstm_units=int(exp_lstm_units),
                learning_rate=float(exp_lr),
                epochs=int(exp_epochs),
                current_batch_size=int(exp_batch_size),
                n_lstm_layers=int(exp_n_layers),
                stateful=bool(exp_stateful),
                features_to_use=exp_features_to_use,
            )

            if result is None:
                status.write("")
                progress.progress(0.0)
                st.error("Experiment training failed or not enough data.")
            else:
                final_val_loss, model_path, bias_path = result
                progress.progress(1.0)
                status.write("Experiment finished.")

                # Present validation loss in scientific notation and show only
                # the model filename (not the full path) in the status message.
                st.success(
                    f"Experiment val_loss={final_val_loss:.3e}, "
                    f"model={os.path.basename(model_path)}"
                )

                # Append to in-memory experiments table in session_state.
                # Store validation loss as a scientific-notation string and only
                # the filenames for the model and bias-correction artifacts.
                row = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "frequency": exp_freq,
                    "tsteps": exp_tsteps,
                    "lstm_units": int(exp_lstm_units),
                    "batch_size": int(exp_batch_size),
                    "learning_rate": float(exp_lr),
                    "n_lstm_layers": int(exp_n_layers),
                    "stateful": bool(exp_stateful),
                    "features_to_use": ",".join(exp_features_to_use),
                    "epochs": int(exp_epochs),
                    "validation_loss": f"{float(final_val_loss):.3e}",
                    "model": os.path.basename(model_path),
                    "correction_path": os.path.basename(bias_path),
                }
                # Backwards-compatible: keep legacy session_state list, but
                # treat ui_state.experiments.runs as the primary store.
                st.session_state.setdefault("lstm_experiments", []).append(row)

                runs: list[dict] = exp_state.get("runs", [])
                runs.append(row)
                if len(runs) > MAX_HISTORY_ROWS:
                    runs = runs[-MAX_HISTORY_ROWS:]
                exp_state["runs"] = runs
                save_json_history("experiments_runs.json", runs)
        except Exception as exc:  # pragma: no cover - UI convenience
            progress.progress(0.0)
            status.write("")
            st.error(f"Experiment failed: {exc}")

    # Experiments table with an action to load a row into the Train tab.
    exp_rows = exp_state.get("runs") or st.session_state.get("lstm_experiments", [])
    
    def load_experiment_to_training(experiment: dict) -> None:
        """Callback to load selected experiment into training tab."""
        st.session_state["train_prefill"] = experiment
        # Mirror into ui_state.training so the Train tab can prefer it.
        train_state = get_ui_state().setdefault("training", {})
        train_state["train_prefill"] = experiment
        st.success("Loaded experiment into Train & Promote tab. Switch to that tab to train fully.")
    
    # Professional experiment table with integrated selection
    components.render_experiment_table(
        st,
        pd,
        experiments=exp_rows,
        on_select=load_experiment_to_training
    )
