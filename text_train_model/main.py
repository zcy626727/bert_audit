
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,Trainer,DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("./text_train_model/Erlangshen-Roberta-110M-Sentiment/")
model = AutoModelForSequenceClassification.from_pretrained("./text_train_model/Erlangshen-Roberta-110M-Sentiment/")

# 数据集
train_dataset = load_dataset( "csv",data_dir='./text_train_model/dataset/' , data_files= {'train':['train.csv','my_train.csv']} )
test_dataset = load_dataset( "csv" , data_files= {'test':'./text_train_model/dataset/test.csv'} )

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./save_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=test_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

tokenizer.save_pretrained("./text_model/")
model.save_pretrained("./text_model/")
