"""What kind of Kazakh errors remain? Categorizes word errors of saved runs.

Reads the per-utterance CSVs written by 01_zero_shot.py, so it needs no GPU and no model.
Usage: python error_analysis_kk.py
"""
import json
import re

import jiwer
import pandas as pd

RUNS = {
    "whisper-small zero-shot": "results/phase1/whisper-small__kk.csv",
    "whisper-small fine-tuned": "results/phase3/whisper-small-kk-final__kk.csv",
    "whisper-large-v3-turbo": "results/phase1/whisper-large-v3-turbo__kk.csv",
    "mms-1b-all": "results/phase1/mms-1b-all__kk.csv",
}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# Kazakh-specific letters and the Russian letters they look and sound closest to
KK_RU_PAIRS = {("ә", "а"), ("ғ", "г"), ("қ", "к"), ("ң", "н"), ("ө", "о"),
               ("ұ", "у"), ("ү", "у"), ("һ", "х"), ("і", "и")}


def kazakh_letter_confusion(ref_word, hyp_word):
    """True if the two words differ only by Kazakh-specific letters replaced with Russian look-alikes."""
    if len(ref_word) != len(hyp_word) or ref_word == hyp_word:
        return False
    diffs = [(a, b) for a, b in zip(ref_word, hyp_word) if a != b]
    return all((a, b) in KK_RU_PAIRS or (b, a) in KK_RU_PAIRS for a, b in diffs)


def classify(ref_word, hyp_word):
    """Bucket one substitution."""
    if re.fullmatch(r"\d+", ref_word) or re.fullmatch(r"\d+", hyp_word):
        return "number format (digits vs words)"
    if re.search(r"[a-z]", ref_word) or re.search(r"[a-z]", hyp_word):
        return "latin-script word"
    if kazakh_letter_confusion(ref_word, hyp_word):
        return "Kazakh letter replaced by Russian look-alike"
    d = edit_distance(ref_word, hyp_word)
    if d == 1:
        return "one character off (other)"
    if common_prefix(ref_word, hyp_word) >= 0.6 * len(ref_word) and d <= 4:
        return "same stem, different ending"
    return "different word"


out = {}
for name, path in RUNS.items():
    d = pd.read_csv(path, keep_default_na=False)
    subs, dels, ins, buckets = 0, 0, 0, {}
    for ref, hyp in zip(d.ref_norm, d.hyp_norm):
        o = jiwer.process_words(ref, hyp)
        subs += o.substitutions
        dels += o.deletions
        ins += o.insertions
        r_words, h_words = ref.split(), hyp.split()
        for chunk in o.alignments[0]:
            if chunk.type == "substitute":
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                                range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    b = classify(r_words[i], h_words[j])
                    buckets[b] = buckets.get(b, 0) + 1
    total = subs + dels + ins
    exact = float((d.ref_norm == d.hyp_norm).mean())
    # identical once spaces are ignored: the words are right, only the boundaries are wrong
    spaceless = float((d.ref_norm.str.replace(" ", "") == d.hyp_norm.str.replace(" ", "")).mean())
    out[name] = {
        "total_word_errors": total,
        "substitutions_%": round(100 * subs / total, 1),
        "deletions_%": round(100 * dels / total, 1),
        "insertions_%": round(100 * ins / total, 1),
        "substitution_types_%": {k: round(100 * v / subs, 1)
                                 for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "utterances_exactly_correct_%": round(100 * exact, 1),
        "utterances_correct_ignoring_word_boundaries_%": round(100 * spaceless, 1),
    }
    print(f"\n## {name}")
    print(json.dumps(out[name], indent=2, ensure_ascii=False))

json.dump(out, open("results/error_analysis_kk.json", "w"), indent=2, ensure_ascii=False)
print("\nwrote results/error_analysis_kk.json")
