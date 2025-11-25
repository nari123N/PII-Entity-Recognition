import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import re
from typing import List, Tuple, Dict

# Conservative regex checks for high-risk PII types
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_DIGITS_RE = re.compile(r"\d")
CREDIT_DIGITS_RE = re.compile(r"(?:\d[ -]*){12,19}")

def normalize_spoken_span(s: str) -> str:
    """Convert simple spoken forms into text to help regex-based verification."""
    s = s.replace(" at ", "@").replace(" dot ", ".")
    s = s.replace(" period ", ".").replace(" underscore ", "_")
    s = s.replace(" oh ", " 0 ").replace(" zero ", "0")
    return " ".join(s.split())

def verify_span_label(span_text: str, label: str) -> bool:
    """Return True if span_text looks plausibly like `label`. Conservative checks."""
    st = normalize_spoken_span(span_text.lower())
    if label == "EMAIL":
        return EMAIL_RE.search(st) is not None
    if label == "PHONE":
        digits = re.sub(r"\D", "", st)
        # allow local/intl range
        return 7 <= len(digits) <= 15
    if label == "CREDIT_CARD":
        digits = re.sub(r"\D", "", st)
        return 12 <= len(digits) <= 19
    return True

def build_id2label_from_model_or_fallback(model) -> Dict[int, str]:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "id2label"):
        try:
            return {int(k): v for k, v in cfg.id2label.items()}
        except Exception:
            # some configs already have int keys
            return {int(k): v for k, v in enumerate(cfg.id2label.values())}
    # fallback to imported mapping (strings assumed)
    # ensure keys are ints
    return {int(k): v for k, v in ID2LABEL.items()}

def bio_to_spans(text: str, offsets: List[Tuple[int, int]], label_ids: List[int], id2label: Dict[int, str]):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # skip special tokens or empty offsets
        if start == 0 and end == 0:
            continue
        label = id2label.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None
            continue

        # handle possible malformed labels robustly
        if "-" not in label:
            # treat as O if format unexpected
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                # unexpected I- for different entity: close previous and start new span
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
        else:
            # unknown prefix - close current
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    # clip spans to text bounds and remove degenerate
    tlen = len(text)
    filtered = []
    for s, e, lab in spans:
        s = max(0, min(s, tlen))
        e = max(0, min(e, tlen))
        if s < e:
            filtered.append((s, e, lab))
    return filtered

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1, help="Torch intra-op threads")
    ap.add_argument("--drop_pii_when_verify_fails", action="store_true",
                    help="Drop EMAIL/PHONE/CREDIT_CARD spans that fail simple regex checks")
    return ap.parse_args()

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    id2label = build_id2label_from_model_or_fallback(model)

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            uid = obj.get("id", None)

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids, id2label)
            ents = []
            for s, e, lab in spans:
                span_text = text[int(s):int(e)]
                if args.drop_pii_when_verify_fails and lab in ("EMAIL", "PHONE", "CREDIT_CARD"):
                    if not verify_span_label(span_text, lab):
                        # skip this prediction to increase precision
                        continue
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })

            results[uid] = ents

    # write mapping id -> list of ents
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()
