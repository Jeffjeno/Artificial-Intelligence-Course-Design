import optuna
from scripts.train import train_xlmr

def objective(trial):
    """
    Objective function for hyperparameter optimization using Optuna.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    accuracy = train_xlmr(learning_rate=learning_rate, batch_size=batch_size, dropout_rate=dropout_rate, return_accuracy=True)
    return accuracy

def optimize_hyperparameters(file_path):
    """
    Optimize hyperparameters using Bayesian optimization (Optuna).
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)