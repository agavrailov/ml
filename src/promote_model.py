import os
import json
import sys
from datetime import datetime # Moved import to top

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_latest_model_path, ACTIVE_MODEL_PATH_FILE, MODEL_REGISTRY_DIR

def promote_model():
    """
    Finds the latest model in the registry and promotes it to be the active model
    by writing its path to the active_model.txt file.
    """
    print("Attempting to promote the latest model...")

    # 1. Find the latest model
    latest_model_path = get_latest_model_path()

    if latest_model_path is None:
        print(f"Error: No models found in the registry at '{MODEL_REGISTRY_DIR}'.")
        print("Please train a model first using 'python src/train.py'.")
        return

    print(f"Found latest model: {latest_model_path}")

    # 2. Create the content for the active model file
    active_model_info = {
        "path": latest_model_path,
        "promoted_at": datetime.now().isoformat()
    }

    # 3. Write the path to the active_model.txt file
    try:
        os.makedirs(os.path.dirname(ACTIVE_MODEL_PATH_FILE), exist_ok=True)
        with open(ACTIVE_MODEL_PATH_FILE, 'w') as f:
            json.dump(active_model_info, f, indent=4)
        print(f"Successfully promoted model. Active model pointer created at: {ACTIVE_MODEL_PATH_FILE}")
    except Exception as e:
        print(f"Error: Failed to write to active model file at '{ACTIVE_MODEL_PATH_FILE}'.")
        print(f"Details: {e}")

if __name__ == "__main__":
    from datetime import datetime
    promote_model()
