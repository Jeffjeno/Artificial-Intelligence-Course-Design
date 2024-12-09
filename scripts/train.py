import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from scripts.preprocess import preprocess_data

def train_xlmr(file_path, learning_rate=2e-5, batch_size=16, dropout_rate=0.1, return_accuracy=False):
    """
    Trains the XLM-RoBERTa model with the given hyperparameters.
    """
    model_name = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.config.hidden_dropout_prob = dropout_rate
    
    dataset = load_dataset("csv", data_files={"train": file_path})
    dataset = dataset["train"].train_test_split(test_size=0.2)
    dataset = dataset.map(lambda batch: preprocess_data(batch, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    if return_accuracy:
        eval_results = trainer.evaluate()
        return eval_results["eval_accuracy"]