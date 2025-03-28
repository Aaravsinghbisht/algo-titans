import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

import torch 

PRETRAINED_MODEL = ""
MODEL_NAME = "Donovin"
DATASET_JSON = ""
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 10
MAX_INPUT_LENTH = 150
MAX_OUTPUT_LENTH = 256
LEARNING_RATE = 2e-5

def load_chat_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        df = pd.read_json(f)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' fields.")
    return Dataset.from_pandas(df)


def preprocess_function(examples, tokenizer):
    inputs = ["question: " + ins + " \n answer:" for ins in examples["question"]]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_OUTPUT_LENTH, truncation=True)
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
    trainer.save(f"./{MODEL_NAME}")
    print(f"trainning succesfull. model save to ./{MODEL_NAME}_module")

if __name__ == "__main__":
    main()