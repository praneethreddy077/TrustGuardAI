import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings

class TextFakeNewsDetector:
    """
    If fine-tuned weights exist at settings.TEXT_FINETUNED_WEIGHTS, load them.
    Otherwise fall back to SST-2 sentiment head (class 1 ~ 'real' proxy).
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL_HF)
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.TEXT_MODEL_HF)
        if os.path.exists(settings.TEXT_FINETUNED_WEIGHTS):
            sd = torch.load(settings.TEXT_FINETUNED_WEIGHTS, map_location="cpu")
            try:
                self.model.load_state_dict(sd, strict=False)
            except Exception:
                pass
        self.model.eval()

    def predict_proba(self, text: str):
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        # assume index 1 corresponds to REAL class in our finetune; on SST-2 it's 'positive'
        return {"p_fake": float(probs[0]), "p_real": float(probs[1])}

    def predict(self, text: str):
        proba = self.predict_proba(text)
        label = "REAL" if proba["p_real"] >= proba["p_fake"] else "FAKE"
        conf  = max(proba["p_real"], proba["p_fake"])
        return {"label": label, "probability": float(conf), **proba}
