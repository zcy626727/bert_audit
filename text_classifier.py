import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,Trainer,DataCollatorWithPadding
from datasets import load_dataset


class TextClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoint-3218/")
        self.model = AutoModelForSequenceClassification.from_pretrained("./checkpoint-3218/")
        self.model.eval()

    def predict(self, text:str):
        model_input = self.tokenizer(text,return_tensors="pt",padding=True)
        model_output = self.model(**model_input, return_dict=False)
        prediction = torch.argmax(model_output[0].cpu(), dim=-1)
        prediction = [p.item() for p in prediction]
        return prediction

