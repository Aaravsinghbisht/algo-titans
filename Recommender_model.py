import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch

PRETRAINED_MODEL = "t5-small"
MODEL_NAME = "AreaRecommender"
DATASET_JSON = "resume_analysis_results.json"
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 10
MAX_INPUT_LENGTH = 200
MAX_OUTPUT_LENGTH = 256
LEARNING_RATE = 2e-5

def load_chat_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        df = pd.read_json(f)
    required_fields = ["category", "resume_quality", "weaknesses", "improvements"]
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Dataset must contain these fields: {', '.join(required_fields)}. Found columns: {df.columns.tolist()}")
    return Dataset.from_pandas(df)

def preprocess_function(examples, tokenizer):
    inputs = [
        f"Category: {cat}\nResume Quality: {quality}\nWeaknesses: {weakness}\nImprovements:"
        for cat, quality, weakness in zip(examples["category"], examples["resume_quality"], examples["weaknesses"])
    ]
    targets = examples["improvements"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_OUTPUT_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    dataset = load_chat_dataset(DATASET_JSON)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    tokenize_fn = lambda examples: preprocess_function(examples, tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{MODEL_NAME}_model",
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=EPOCHS,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(f"./{MODEL_NAME}")
    print(f"Training successful. Model saved to ./{MODEL_NAME}_model")

if __name__ == "__main__":
    main()
