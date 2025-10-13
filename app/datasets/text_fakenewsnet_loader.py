import pandas as pd
from pathlib import Path

def load_fakenewsnet(root="data/text"):
    """
    Expects:
      data/text/buzzfeed/fake/*_content.csv
      data/text/buzzfeed/real/*_content.csv
      data/text/politifact/fake/*_content.csv
      data/text/politifact/real/*_content.csv
    Returns a DataFrame with columns: ['source','label','title','text']
    """
    root = Path(root)
    frames = []
    for src in ("buzzfeed", "politifact"):
        for lab in ("fake", "real"):
            files = list((root / src / lab).glob("*_content*.csv"))
            for f in files:
                df = pd.read_csv(f, encoding="utf-8", engine="python")
                # try common content columns
                if "text" not in df.columns:
                    for c in ["content", "body", "article", "news_text"]:
                        if c in df.columns:
                            df = df.rename(columns={c: "text"}); break
                if "title" not in df.columns:
                    for c in ["headline", "title_text"]:
                        if c in df.columns:
                            df = df.rename(columns={c: "title"}); break
                # keep only text/title if present
                keep = [c for c in ["title","text"] if c in df.columns]
                df = df[keep].copy()
                df["source"] = src
                df["label"] = 0 if lab == "fake" else 1
                frames.append(df)
    if not frames:
        raise FileNotFoundError("No FakeNewsNet CSVs found under data/text/**")
    out = pd.concat(frames, ignore_index=True).dropna()
    # basic cleanup
    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["text"].str.len() > 20]
    return out.reset_index(drop=True)
