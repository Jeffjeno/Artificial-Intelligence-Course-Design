import os
from scripts.train import train_xlmr
from scripts.seq2seq import train_seq2seq
from scripts.optimization import optimize_hyperparameters
from scripts.clustering import cluster_data

# Paths
data_dir = './data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

# Step 1: Preprocessing and Clustering
print("Clustering data for balance...")
cluster_data(train_file)

# Step 2: Hyperparameter Optimization
print("Optimizing hyperparameters...")
optimize_hyperparameters(train_file)

# Step 3: Train XLM-RoBERTa Model
print("Training XLM-RoBERTa model...")
train_xlmr(train_file)

# Step 4: Train Seq2Seq Model
print("Training Seq2Seq model...")
train_seq2seq(train_file)

print("All training processes completed.")