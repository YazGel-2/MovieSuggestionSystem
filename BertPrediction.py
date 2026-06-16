import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "./BertModel"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

model.eval()

def predict_sentiment_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred_id = torch.argmax(logits, dim=-1).item()
    label = label_encoder.inverse_transform([pred_id])[0]

    return label

print(predict_sentiment_bert("This movie was absolutely amazing!"))
print(predict_sentiment_bert("Worst film I have ever seen"))