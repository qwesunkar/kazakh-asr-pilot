"""Shared helpers: FLEURS loading, text normalization, metrics.

All phases import from here so that data and normalization are identical everywhere.
"""
import io
import re
import unicodedata

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

# Short code -> FLEURS config, Whisper language name, MMS adapter code
LANGS = {
    "kk": {"fleurs": "kk_kz", "whisper": "kazakh", "mms": "kaz"},
    "ru": {"fleurs": "ru_ru", "whisper": "russian", "mms": "rus"},
    "en": {"fleurs": "en_us", "whisper": "english", "mms": "eng"},
}


def normalize(text: str) -> str:
    """NFKC, lowercase, ё -> е, punctuation/symbols -> space, collapse whitespace.

    Like Whisper's BasicTextNormalizer, but it does NOT delete text inside
    brackets (about 10% of FLEURS references contain spoken words in parentheses)
    and it keeps combining marks (no diacritic stripping). ё -> е because FLEURS
    Russian references use both spellings inconsistently.
    """
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load_fleurs(lang: str, split: str = "test", limit: int | None = None):
    """Return a list of dicts: id, audio (float32 16 kHz mono), ref (raw transcription)."""
    ds = load_dataset("google/fleurs", LANGS[lang]["fleurs"], split=split)
    ds = ds.cast_column("audio", Audio(decode=False))  # decode WAV bytes ourselves, no torchcodec
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for ex in ds:
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        assert sr == 16000 and wav.ndim == 1, (sr, wav.shape)
        items.append({"id": ex["id"], "audio": wav, "ref": ex["raw_transcription"]})
    return items


KSC_TEST = ("hf://datasets/Shirali/ISSAI_KSC_335RS_v_1_1/data/"
            "test-00000-of-00001-594d52976822c6c4.parquet")


def load_ksc(split: str = "test", limit: int | None = None):
    """Kazakh Speech Corpus (ISSAI, CC BY 4.0) test split, read from a community parquet mirror.

    Out-of-domain check for models tuned on FLEURS. Same item format as load_fleurs().
    References are already lowercase, unpunctuated, and spell numbers out as words.
    """
    assert split == "test", "only the KSC test split is used here"
    ds = load_dataset("parquet", data_files={"test": KSC_TEST}, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for ex in ds:
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        assert sr == 16000 and wav.ndim == 1, (sr, wav.shape)
        items.append({"id": ex["uttID"], "audio": wav, "ref": ex["text"]})
    return items


def score(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level WER/CER (total edits / total reference words or chars) on normalized text."""
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    assert all(r), "empty reference after normalization"
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h)}


def bootstrap_ci(refs: list[str], hyps: list[str], n_resamples: int = 1000, seed: int = 0) -> dict:
    """95% bootstrap confidence interval for corpus WER/CER, resampling utterances with replacement.

    Corpus WER is a ratio of sums, so each resample re-adds both the errors and the reference
    length of the drawn utterances; this captures how much the score depends on which
    utterances happen to be in the test set.
    """
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    w_err, w_len, c_err, c_len = [], [], [], []
    for ref, hyp in zip(r, h):
        o = jiwer.process_words(ref, hyp)
        w_err.append(o.substitutions + o.deletions + o.insertions)
        w_len.append(len(ref.split()))
        o = jiwer.process_characters(ref, hyp)
        c_err.append(o.substitutions + o.deletions + o.insertions)
        c_len.append(len(ref))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_resamples, len(r)))
    out = {}
    for name, err, length in [("wer", w_err, w_len), ("cer", c_err, c_len)]:
        err, length = np.asarray(err), np.asarray(length)
        ratios = err[idx].sum(1) / length[idx].sum(1)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = float(lo), float(hi)
    return out


def paired_bootstrap_diff(refs: list[str], hyps_a: list[str], hyps_b: list[str],
                          n_resamples: int = 1000, seed: int = 0) -> dict:
    """95% CI for WER(b) - WER(a) on the same utterances (paired bootstrap).

    Both systems are scored on every resample of the same utterance draw, so the shared
    difficulty of the utterances cancels out. A CI that excludes 0 means the systems differ.
    """
    r = [normalize(x) for x in refs]
    counts = {}
    for key, hyps in [("a", hyps_a), ("b", hyps_b)]:
        err, length = [], []
        for ref, hyp in zip(r, [normalize(x) for x in hyps]):
            o = jiwer.process_words(ref, hyp)
            err.append(o.substitutions + o.deletions + o.insertions)
            length.append(len(ref.split()))
        counts[key] = (np.asarray(err), np.asarray(length))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_resamples, len(r)))
    wer = {k: e[idx].sum(1) / n[idx].sum(1) for k, (e, n) in counts.items()}
    diff = wer["b"] - wer["a"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    point = counts["b"][0].sum() / counts["b"][1].sum() - counts["a"][0].sum() / counts["a"][1].sum()
    return {"diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def audio_seconds(items) -> float:
    return float(sum(len(x["audio"]) for x in items) / 16000)


def length_sorted(items):
    """Indices sorted by audio length (longest first) to reduce padding in batches."""
    return list(np.argsort([-len(x["audio"]) for x in items], kind="stable"))
