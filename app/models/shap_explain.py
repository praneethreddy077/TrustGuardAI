# SHAP over transformer outputs (lightweight KernelExplainer)
import shap, numpy as np
from app.models.text_model import TextFakeNewsDetector

def shap_top_words(text: str, k: int = 15):
    model = TextFakeNewsDetector()

    def f(txts):
        # returns probability of REAL class
        import numpy as np
        return np.array([model.predict_proba(t)["p_real"] for t in txts])

    explainer = shap.Explainer(f, shap.maskers.Text())
    sv = explainer([text])
    # Build (token, shap_value) list
    toks = sv.data[0]
    vals = sv.values[0]
    pairs = list(zip(toks, vals))
    # rank by absolute contribution
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]
