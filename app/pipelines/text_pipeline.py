from app.datasets.text_fakenewsnet_loader import load_fakenewsnet
from app.datasets.preprocess_utils import clean_text
from app.models.text_explain import TextFakeNewsDetector
from app.models.shap_explain import shap_top_words

def analyze_text_sample(text: str):
    text = clean_text(text)
    model = TextFakeNewsDetector()
    pred = model.predict(text)
    top = shap_top_words(text, k=10)
    return {"prediction": pred, "top_words": top}

def sample_dataset_row(idx: int = 0):
    df = load_fakenewsnet()
    row = df.iloc[idx]
    row_text = clean_text(row.get("text", ""))
    return {"source": row["source"], "label": int(row["label"]), "text": row_text}
