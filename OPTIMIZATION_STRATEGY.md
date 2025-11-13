# Strategy for Optimizing and Evaluating LSTM Model Results

Now that the refactoring is complete and the `experiment_runner.py` is ready, we can systematically optimize and evaluate the LSTM model. The goal is to achieve an MAE below 10 and a high correlation between actual and predicted data series.

## Phase 1: Initial Exploration and Baseline Establishment

1.  **Run a Small Subset of Experiments:**
    *   Start with a limited set of hyperparameters to get a baseline. For example, pick one or two `RESAMPLE_FREQUENCIES`, a few `TSTEPS_OPTIONS`, and the default `LSTM_UNITS`, `BATCH_SIZE`, `N_LSTM_LAYERS=1`, `STATEFUL=True`, `OPTIMIZER='rmsprop'`, `LOSS_FUNCTION='mae'`, and `FEATURES_TO_USE_OPTIONS[0]`.
    *   This will confirm the `experiment_runner.py` works as expected and provide initial performance metrics.

2.  **Analyze Baseline Results:**
    *   Examine `experiment_results.json` and `best_hyperparameters.json`.
    *   Note the MAE and correlation for the initial runs. This will be our baseline to beat.
    *   Review `evaluation_plot.png` for visual inspection of predictions vs. actuals.

## Phase 2: Iterative Hyperparameter Tuning

This phase involves systematically exploring the hyperparameter space. It's crucial to manage the computational cost, so we'll use a structured approach.

1.  **Focus on One or Two Key Parameters at a Time (or small groups):**
    *   **Frequency and TSTEPS:** These often have the biggest impact. Experiment with all `RESAMPLE_FREQUENCIES` and `TSTEPS_OPTIONS` first, keeping other parameters at their default or best-performing values from Phase 1.
    *   **LSTM Units and Layers:** Once a promising (frequency, tsteps) combination is found, vary `LSTM_UNITS_OPTIONS` and `N_LSTM_LAYERS_OPTIONS`.
    *   **Batch Size:** Experiment with `BATCH_SIZE_OPTIONS`.
    *   **Dropout Rates:** Test `DROPOUT_RATE_OPTIONS` to prevent overfitting.
    *   **Features:** Experiment with `FEATURES_TO_USE_OPTIONS`. This might require re-running `src/data_processing.py` for each feature set if new features are introduced.
    *   **Optimizer and Loss Function:** Explore `OPTIMIZER_OPTIONS` and `LOSS_FUNCTION_OPTIONS`.

2.  **Implement a Search Strategy:**
    *   **Grid Search (Current Approach):** The `experiment_runner.py` currently performs a grid search. For a large number of parameters, this can be computationally expensive. Start with a coarse grid, then refine.
    *   **Random Search:** If the grid search becomes too slow, consider implementing a random search within defined ranges for each parameter. This can often find good results faster than a full grid search.
    *   **Bayesian Optimization (Advanced):** For more advanced tuning, consider integrating libraries like `KerasTuner` (which was mentioned in the context) or `Optuna` for more intelligent search strategies.

3.  **Evaluation Metrics for Optimization:**
    *   **Primary Metric: MAE:** The main objective is to get MAE below 10.
    *   **Secondary Metric: Correlation:** Aim for a high Pearson correlation coefficient (close to 1) between actual and predicted values.
    *   **Validation Loss:** Monitor this during training to ensure the model is learning and not overfitting.
    *   **Standard Deviation of Predictions vs. Actuals:** As observed before, a large discrepancy here indicates underfitting.

4.  **Early Stopping:** Ensure early stopping is enabled during training to prevent overfitting and save computational resources.

## Phase 3: Architectural Experimentation (if needed)

If hyperparameter tuning alone doesn't yield the desired results, consider more fundamental changes to the model architecture.

1.  **Bidirectional LSTMs:** Modify `src/model.py` to include an option for `Bidirectional` LSTM layers.
2.  **GRU Layers:** Add an option to use GRU layers instead of LSTMs.
3.  **Stacked LSTMs with different units:** Experiment with different `lstm_units` for each layer in a stacked LSTM.
4.  **CNN-LSTM:** Integrate a 1D Convolutional layer before the LSTM layers for local feature extraction.
5.  **Attention Mechanisms:** Explore adding an attention layer to the model.

## Phase 4: Data Enhancement (if needed)

If model architecture and hyperparameters are exhausted, look at the data itself.

1.  **More Features:** Introduce more sophisticated technical indicators (e.g., MACD, Bollinger Bands, Stochastic Oscillator, ADX).
2.  **External Data:** Incorporate relevant external data (e.g., news sentiment, economic indicators, related asset prices).
3.  **Feature Engineering:** Create custom features that might capture specific market dynamics.

## Workflow for Running Experiments:

1.  **Define Experiment Scope:** Decide which parameters or architectural changes to test in a given batch of experiments.
2.  **Update `src/config.py` (if necessary):** For global defaults or ranges.
3.  **Run `src/experiment_runner.py`:** Execute the script.
4.  **Review Results:** Analyze `experiment_results.json` and `best_hyperparameters.json`.
5.  **Visualize:** Generate plots for the best-performing models using `src/evaluate_model.py` (or integrate plotting into `experiment_runner.py` for automated visualization of top performers).
6.  **Iterate:** Based on the results, refine the parameter ranges or choose new parameters to explore.

## Important Considerations:

*   **Computational Resources:** Running many experiments can be resource-intensive. Start small and scale up.
*   **Reproducibility:** Ensure random seeds are consistently set for fair comparisons.
*   **Version Control:** Commit changes regularly, especially after significant refactoring or when a promising model configuration is found.
*   **Logging:** Ensure `experiment_runner.py` logs enough detail to trace back the exact configuration of each experiment.

## Addressing New Requirements:

*   **Summary at Major Milestones:**
    *   The `experiment_runner.py` will print a summary after each experiment run, including the experiment ID, hyperparameters, validation loss, MAE, and correlation.
    *   It will also provide a final summary of the best-performing models across all frequencies/TSTEPS combinations.
*   **Load Parameters of a Certain Test:**
    *   The `experiment_results.json` will store all experiment details, including the full set of hyperparameters.
    *   A new function will be added to `src/experiment_runner.py` (or a separate utility script) to load a specific experiment's parameters by its ID and then re-run the training/evaluation for that configuration.
*   **Each Test Should Have an ID:**
    *   Each experiment run will be assigned a unique ID (e.g., a timestamp combined with a hash of its hyperparameters, or simply an incrementing counter). This ID will be stored in `experiment_results.json` and used for tracking.
