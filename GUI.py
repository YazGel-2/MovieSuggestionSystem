import tkinter as tk
from tkinter import ttk, messagebox
from BertPrediction import predict_sentiment_bert
from DistilRobertaPrediction import predict_sentiment_roberta

model_map = {
    "BERT": predict_sentiment_bert,
    "DistilRoBERTa": predict_sentiment_roberta
}

model_metrics = {
    "BERT": {"accuracy": 0.68, "precision": 0.68, "recall": 0.57, "f1": 0.60},
    "DistilRoBERTa": {"accuracy": 0.70, "precision": 0.70, "recall": 0.70, "f1": 0.70},
}


def update_metrics(event=None):
    model = combo.get()
    if model in model_metrics:
        m = model_metrics[model]

        acc_label.config(text=f"Accuracy: {m['accuracy']:.2f}")
        prec_label.config(text=f"Precision: {m['precision']:.2f}")
        rec_label.config(text=f"Recall: {m['recall']:.2f}")
        f1_label.config(text=f"F1 Score: {m['f1']:.2f}")

def run_prediction():
    text = text_box.get("1.0", tk.END).strip()
    model = combo.get()

    if not text:
        messagebox.showerror("Hata", "Metin boş!")
        return

    result = model_map[model](text)
    result_label.config(text=result)

root = tk.Tk()
root.title("Sentiment Dashboard")
root.geometry("750x550")
root.configure(bg="#1e1e2f")

# Header
tk.Label(
    root,
    text="Sentiment Analysis Dashboard",
    font=("Arial", 18, "bold"),
    bg="#1e1e2f",
    fg="white"
).pack(pady=10)

# Card
card = tk.Frame(root, bg="#2a2a40")
card.pack(padx=20, pady=10, fill="both")

# Model dropdown
tk.Label(card, text="Model Seç", bg="#2a2a40", fg="white").pack()

combo = ttk.Combobox(card, values=list(model_map.keys()), state="readonly", width=40)
combo.pack(pady=5)
combo.bind("<<ComboboxSelected>>", update_metrics)

# Metrics panel
metrics_frame = tk.Frame(card, bg="#2a2a40")
metrics_frame.pack(pady=10)

acc_label = tk.Label(metrics_frame, text="Accuracy:", fg="white", bg="#2a2a40")
acc_label.grid(row=0, column=0, padx=10)

prec_label = tk.Label(metrics_frame, text="Precision:", fg="white", bg="#2a2a40")
prec_label.grid(row=0, column=1, padx=10)

rec_label = tk.Label(metrics_frame, text="Recall:", fg="white", bg="#2a2a40")
rec_label.grid(row=1, column=0, padx=10)

f1_label = tk.Label(metrics_frame, text="F1 Score:", fg="white", bg="#2a2a40")
f1_label.grid(row=1, column=1, padx=10)

update_metrics()

# Text input
tk.Label(card, text="Film Yorumu", bg="#2a2a40", fg="white").pack(pady=5)

text_box = tk.Text(card, height=6, width=60, bg="#1e1e2f", fg="white", insertbackground="white")
text_box.pack()

# Button
tk.Button(
    card,
    text="Analiz Et",
    command=run_prediction,
    bg="#4c7dff",
    fg="white"
).pack(pady=10)

# Result
result_label = tk.Label(
    root,
    text="Sonuç bekleniyor...",
    font=("Arial", 14, "bold"),
    fg="#00ff99",
    bg="#1e1e2f"
)
result_label.pack(pady=15)

root.mainloop()