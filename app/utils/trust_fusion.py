# app/utils/trust_fusion.py
def fuse_scores(text_prob=None, image_prob=None, w_text=0.5, w_image=0.5):
    """
    text_prob, image_prob are REAL-class confidences in [0,1].
    Returns fused score in [0,1]. If only one is available, returns that one.
    """
    vals, wts = [], []
    if text_prob is not None:
        vals.append(float(text_prob)); wts.append(float(w_text))
    if image_prob is not None:
        vals.append(float(image_prob)); wts.append(float(w_image))
    if not vals:
        return 0.5
    if len(vals) == 1:
        return round(vals[0], 3)
    wsum = sum(wts) if sum(wts) else 1.0
    return round(sum(v*w for v,w in zip(vals,wts)) / wsum, 3)
