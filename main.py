from flask import Flask, request, render_template
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)


MODEL_PATH = "model/model_clinicalbert_top50new/"
DESC_PATH = "model/dataset_translated.csv"


# LOAD MODEL & TOKENIZER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# Ambil mapping langsung dari model
id2label = model.config.id2label

print("Contoh mapping dari model:", list(id2label.items())[:5])

# LOAD DESKRIPSI ICD

df_desc = pd.read_csv(DESC_PATH)

icd_desc = dict(zip(
    df_desc['icd10_code'],
    df_desc['icd10_desc']
))


# FUNCTION PREDICT

def predict(text):

    text = text.lower().strip()

    if not text or len(text) < 3:
        return []

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probs = torch.softmax(outputs.logits, dim=1)[0]

  
    topk = torch.topk(probs, 1)

    results = []

    for i in range(len(topk.indices)):
        idx = topk.indices[i].item()
        conf = topk.values[i].item() * 100


        icd_code = id2label[idx]

        icd_name = icd_desc.get(icd_code, "Deskripsi tidak tersedia")

        results.append({
            "code": icd_code,
            "name": icd_name,
            "confidence": f"{conf:.2f}%"
        })

    return results


# ROUTE

@app.route("/", methods=["GET", "POST"])
def home():

    results = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            results = predict(text)

    return render_template("index.html", results=results)


# RUN APP

if __name__ == "__main__":
    app.run(debug=True)