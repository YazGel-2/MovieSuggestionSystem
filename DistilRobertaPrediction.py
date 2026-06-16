from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./RobertaModel"

label_map = {0: "sadness", 1: "joy"}

# 🔥 GLOBAL LOAD (1 kez)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

def predict_sentiment_roberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).item()

    return label_map[preds]