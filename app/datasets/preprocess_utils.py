import re

def clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"http\S+", " ", x)
    x = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x
