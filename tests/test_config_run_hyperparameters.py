import json

from src.config import TRAINING, get_run_hyperparameters


def test_get_run_hyperparameters_treats_null_as_missing(tmp_path):
    best_hps_path = tmp_path / "best_hyperparameters.json"

    # Simulate a partially-written / user-edited tuned hyperparameters file that
    # contains explicit nulls.
    best_hps_path.write_text(
        json.dumps(
            {
                "15min": {
                    "5": {
                        "epochs": None,
                        "stateful": False,
                        "batch_size": 16,
                    }
                }
            }
        )
    )

    hps = get_run_hyperparameters(frequency="15min", tsteps=5, best_hps_path=str(best_hps_path))

    # 1) null should not override TRAINING defaults
    assert hps["epochs"] == TRAINING.epochs

    # 2) non-null tuned values should still override defaults
    assert hps["batch_size"] == 16
    assert hps["stateful"] is False
