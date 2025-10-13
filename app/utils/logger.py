import time, json
from pathlib import Path

LOG = Path("predictions_log.jsonl")

def log_event(event: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.time(), **event}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
