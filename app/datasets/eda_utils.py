from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ---------- TEXT EDA ----------
def df_basic_stats(df: pd.DataFrame):
    out = {}
    out["rows"] = len(df)
    out["cols"] = list(df.columns)
    out["class_balance"] = df["label"].value_counts().to_dict()
    lens = df["text"].astype(str).str.split().apply(len)
    out["word_count_mean"] = float(lens.mean())
    out["word_count_median"] = int(lens.median())
    out["word_count_quantiles"] = {q:int(lens.quantile(q)) for q in [0.1,0.25,0.5,0.75,0.9]}
    return out, lens

def top_tokens(df: pd.DataFrame, k=25, stopwords=None):
    stop = set(stopwords or [])
    bag = Counter()
    for t in df["text"].astype(str).tolist():
        for w in t.lower().split():
            if len(w) <= 2: continue
            if w in stop: continue
            bag[w] += 1
    return bag.most_common(k)

# ---------- IMAGE EDA ----------
def image_file_iter(root: Path, max_per_class=1000):
    count = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg",".jpeg",".png"}:
            yield p
            count += 1
            if count >= max_per_class: break

def image_basic_stats(root="data/image", sample_limit=1000):
    root = Path(root)
    stats = {"real":{}, "fake":{}}
    for label, sub in [("real","pristine"), ("fake","manipulated")]:
        widths, heights, aspects, bright, blur = [], [], [], [], []
        folder = root / sub
        for p in image_file_iter(folder, sample_limit):
            try:
                img = Image.open(p).convert("RGB")
                w,h = img.size
                arr = np.asarray(img)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                widths.append(w); heights.append(h)
                aspects.append(round(w/h, 4))
                bright.append(float(gray.mean()))
                blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            except Exception:
                continue
        stats[label] = {
            "count": len(widths),
            "width_mean": float(np.mean(widths)) if widths else 0,
            "height_mean": float(np.mean(heights)) if heights else 0,
            "aspect_mean": float(np.mean(aspects)) if aspects else 0,
            "brightness_mean": float(np.mean(bright)) if bright else 0,
            "blur_mean": float(np.mean(blur)) if blur else 0,
            "widths": widths, "heights": heights, "aspects": aspects,
            "brightness": bright, "blur": blur
        }
    return stats

def sample_image_paths(root="data/image", n=16):
    root = Path(root)
    real = list((root/"pristine").rglob("*.jpg")) + list((root/"pristine").rglob("*.png"))
    fake = list((root/"manipulated").rglob("*.jpg")) + list((root/"manipulated").rglob("*.png"))
    real = real[: max(0, n//2)]
    fake = fake[: max(0, n - len(real))]
    return [str(p) for p in real+fake]
