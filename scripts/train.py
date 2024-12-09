import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from scripts.preprocess import preprocess_data
from scripts.metrics import compute_metrics

def train_model(train_file):
    # Load dataset
    data = load_dataset('csv', data_files={'train': train_file})
    data = data['train'].train_test_split(test_size=0.2)

    # Tokenizer and Model
    model_name = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Preprocessing
    data = DatasetDict({
        'train': data['train'],
        'val': data['test']
    }).map(lambda batch: preprocess_data(batch, tokenizer), batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()