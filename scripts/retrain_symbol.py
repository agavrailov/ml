"""One-off retrain helper: loads per-symbol hparams from symbol_registry.json and
calls src.train.train_model with them. Monkey-patches optimizer_name / loss_function
so per-symbol loss (e.g. JPM=mae) is honored without requiring a source edit to
train_model. Updates the registry with the resulting model/val_loss."""
from __future__ import annotations
import argparse, json, os, sys

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--frequency", default="60min")
    p.add_argument("--tsteps", type=int, default=5)
    args = p.parse_args()

    from src.model_registry import get_best_model_entry, update_best_model
    entry = get_best_model_entry(args.symbol, args.frequency, args.tsteps)
    if not entry:
        print(f"ERROR: no registry entry for {args.symbol}/{args.frequency}/{args.tsteps}")
        sys.exit(1)
    hp = entry.get("hparams") or {}
    print(f"[retrain] {args.symbol} hparams: {json.dumps(hp, indent=2)}")

    import src.train as _train
    _train.OPTIMIZER_NAME = hp.get("optimizer_name", _train.OPTIMIZER_NAME)
    _train.LOSS_FUNCTION  = hp.get("loss_function",  _train.LOSS_FUNCTION)
    print(f"[retrain] using optimizer_name={_train.OPTIMIZER_NAME} loss_function={_train.LOSS_FUNCTION}")

    result = _train.train_model(
        frequency=args.frequency,
        tsteps=args.tsteps,
        lstm_units=hp.get("lstm_units"),
        learning_rate=hp.get("learning_rate"),
        epochs=hp.get("epochs"),
        current_batch_size=hp.get("batch_size"),
        n_lstm_layers=hp.get("n_lstm_layers"),
        stateful=hp.get("stateful", True),
        features_to_use=hp.get("features_to_use"),
        symbol=args.symbol,
    )
    if result is None:
        print("[retrain] ERROR: train_model returned None")
        sys.exit(2)
    val_loss, model_path, bias_path = result
    print(f"[retrain] val_loss={val_loss:.6e}  model={os.path.basename(model_path)}")

    full_hp = dict(hp)
    full_hp["optimizer_name"] = _train.OPTIMIZER_NAME
    full_hp["loss_function"]  = _train.LOSS_FUNCTION
    wrote = update_best_model(
        symbol=args.symbol,
        frequency=args.frequency,
        tsteps=args.tsteps,
        val_loss=float(val_loss),
        model_path=str(model_path),
        bias_path=str(bias_path) if bias_path else None,
        hparams=full_hp,
        force=True,
    )
    print(f"[retrain] registry updated: {wrote}")

if __name__ == "__main__":
    main()
