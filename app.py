from flask import Flask, request, render_template
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator

app = Flask(__name__)

# =========================================================
# PATH
# =========================================================

MODEL_PATH = "model/model_clinicalbert_top10newww/"
DESC_PATH = "model/dataset_translated.csv"

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# LOAD LABEL MAPPING
# =========================================================

mapping = torch.load(
    os.path.join(MODEL_PATH, "label_mapping.pt"),
    weights_only=False
)

id2label = mapping["id2label"]

label2id = mapping["label2id"]

num_labels = mapping["num_labels"]

print(
    "Contoh mapping:",
    list(id2label.items())[:5]
)

# =========================================================
# CUSTOM MODEL
# =========================================================

class ClinicalBERTClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
        id2label,
        label2id
    ):

        super(
            ClinicalBERTClassifier,
            self
        ).__init__()

        # LOAD BERT
        self.bert = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.bert.config.hidden_size

        # DROPOUT
        self.dropout = nn.Dropout(0.4)

        # RELU
        self.relu = nn.ReLU()

        # CLASSIFIER
        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

        # LABEL CONFIG
        self.config = self.bert.config

        self.config.id2label = id2label

        self.config.label2id = label2id

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS TOKEN
        pooled_output = outputs.last_hidden_state[:, 0]

        pooled_output = self.dropout(
            pooled_output
        )

        pooled_output = self.relu(
            pooled_output
        )

        logits = self.classifier(
            pooled_output
        )

        return logits

# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

# =========================================================
# LOAD MODEL
# =========================================================

model = ClinicalBERTClassifier(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_PATH,
            "model.pth"
        ),
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model berhasil di-load")

# =========================================================
# LOAD ICD DESCRIPTION
# =========================================================

df_desc = pd.read_csv(DESC_PATH)

icd_desc = dict(zip(
    df_desc['icd10_code'],
    df_desc['icd10_desc']
))


# =========================================================
# LOAD ICD XML
# =========================================================

ICD_XML_PATH = "model/icd102019en.xml"

tree = ET.parse(ICD_XML_PATH)
root = tree.getroot()

icd_detail = {}

for cls in root.findall(".//Class"):

    code = cls.get("code")

    data = {
        "title": "",
        "definition": "",
        "note": "",
        "coding_hint": "",
        "exclusions": []
    }

    for rubric in cls.findall("Rubric"):

        kind = rubric.get("kind")

        text = " ".join(
            t.strip()
            for t in rubric.itertext()
            if t.strip()
        )

        if kind == "preferred":
            data["title"] = text

        elif kind == "definition":
            data["definition"] = text

        elif kind == "note":
            data["note"] = text

        elif kind == "coding-hint":
            data["coding_hint"] = text

        elif kind == "exclusion":
            data["exclusions"].append(text)

    icd_detail[code] = data

print("ICD XML Loaded:", len(icd_detail))

# =========================================================
# TRANSLATOR
# =========================================================

# Indonesia -> English
translator_input = GoogleTranslator(
    source='id',
    target='en'
)

# English -> Indonesia
translator_output = GoogleTranslator(
    source='en',
    target='id'
)

# =========================================================
# FUNCTION PREDICT
# =========================================================

def predict(text):

    text = text.lower().strip()

    # VALIDASI INPUT
    if not text or len(text) < 3:
        return []

    # =====================================================
    # TRANSLATE INPUT INDONESIA -> ENGLISH
    # =====================================================

    try:

        translated_text = translator_input.translate(
            text
        )

    except:

        translated_text = text

    # =====================================================
    # TOKENIZER
    # =====================================================

    inputs = tokenizer(
        translated_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs["attention_mask"].to(device)

    # =====================================================
    # PREDICTION
    # =====================================================

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # =====================================================
    # SOFTMAX PROBABILITY
    # =====================================================

    probs = torch.softmax(
        logits,
        dim=1
    )[0]

    # =====================================================
    # ENTROPY
    # =====================================================

    entropy = -torch.sum(
        probs * torch.log(probs + 1e-12)
    ).item()

    # =====================================================
    # UNCERTAINTY LEVEL
    # =====================================================

    if entropy < 0.5:

        uncertainty = "Rendah"

    elif entropy < 1.5:

        uncertainty = "Sedang"

    else:

        uncertainty = "Tinggi"

    # =====================================================
    # TOP 10 PREDICTION
    # =====================================================

    topk = torch.topk(
        probs,
        1
    )

    results = []

    for i in range(len(topk.indices)):

        idx = topk.indices[i].item()

        conf = topk.values[i].item() * 100

        # =================================================
        # ICD CODE
        # =================================================

        icd_code = id2label[idx]
        detail = icd_detail.get(
            icd_code,
            {}
        )
        definition = detail.get(
            "definition",
            "-"
        )
        note = detail.get(
            "note",
            "-"
        )
        coding_hint = detail.get(
            "coding_hint",
            "-"
        )
        exclusions = detail.get(
            "exclusions",
            []
        )

        # =================================================
        # ICD DESCRIPTION ENGLISH
        # =================================================

        icd10_desc_en = icd_desc.get(
            icd_code,
            "Description not available"
        )

        # =================================================
        # ICD DESCRIPTION INDONESIA
        # =================================================

        try:

            icd10_desc_id = translator_output.translate(
                icd10_desc_en
            )

        except:

            icd10_desc_id = icd10_desc_en

        # =================================================
        # RESULT
        # =================================================

        results.append({

            "rank": i + 1,

            "input_asli": text,

            "translated_text": translated_text,

            "code": icd_code,

            "icd10_desc_en": icd10_desc_en,

            "icd10_desc_id": icd10_desc_id,

            "definition": definition,

            "note": note,

            "coding_hint": coding_hint,

            "exclusions": exclusions,

            "confidence": f"{conf:.2f}%",

            "entropy": round(entropy, 4),

            "uncertainty": uncertainty

        })

    return results

# =========================================================
# ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    results = None

    if request.method == "POST":

        text = request.form.get(
            "text",
            ""
        ).strip()

        if text:

            results = predict(text)

    return render_template(
        "index.html",
        results=results
    )

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)