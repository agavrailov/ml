from __future__ import annotations

from src.ui.pages import train_page


def test_append_history_trims_to_max_rows():
    hist = [{"i": 1}, {"i": 2}, {"i": 3}]
    row = {"i": 4}

    out = train_page._append_history(hist, row, max_rows=3)

    assert [r["i"] for r in out] == [2, 3, 4]


def test_build_train_job_request_obj_shapes_and_types():
    req = train_page._build_train_job_request_obj(
        train_freq="15min",
        train_tsteps=10,
        train_lstm_units=64,
        train_lr=0.001,
        train_epochs=5,
        train_batch_size=32,
        train_n_layers=2,
        train_stateful=True,
        train_features_to_use=["Open", "High"],
    )

    assert req["frequency"] == "15min"
    assert req["tsteps"] == 10
    assert req["lstm_units"] == 64
    assert req["learning_rate"] == 0.001
    assert req["epochs"] == 5
    assert req["batch_size"] == 32
    assert req["n_lstm_layers"] == 2
    assert req["stateful"] is True
    assert req["features_to_use"] == ["Open", "High"]
