import numpy as np
import evaluate

def compute_metrics(eval_pred):
    """
    Computes accuracy metrics for model evaluation.
    """
    metric = evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)