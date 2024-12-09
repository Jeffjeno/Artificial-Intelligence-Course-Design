import pandas as pd
import os
from scripts.train import train_model
from scripts.inference import make_predictions

# Paths
data_dir = './data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

# Training
print("Training the model...")
train_model(train_file)

# Inference
print("Running inference on test data...")
test_data = pd.read_csv(test_file)
predictions = make_predictions(test_data)

# Save predictions
submission = pd.DataFrame({'id': test_data['id'], 'prediction': predictions})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")