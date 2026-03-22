from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
import numpy as np

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        examples["highlights"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

small_train = tokenized_dataset["train"].select(range(200))
small_val = tokenized_dataset["validation"].select(range(50))

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-summary-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")