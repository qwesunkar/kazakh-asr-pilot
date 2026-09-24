# Multilingual ASR for low-resource languages: a Kazakh pilot

Sunkar Turmagambetov — MSc Visual Computing, TU Wien
Pilot study for the CVL topic "Multilingual Speech Recognition for Low-Resource Languages"
Code, data and per-utterance outputs: https://github.com/qwesunkar/kazakh-asr-pilot

**Setup.** Kazakh (low-resource, my native language) against Russian and English, everything on one laptop
(RTX 5060 Laptop, 8 GB VRAM). Models: Whisper tiny/base/small/large-v3-turbo and MMS-1b-all. Data: full FLEURS test
splits (kk 856, ru 775, en 647 utterances) and the KSC test split (3334 utterances) as an out-of-domain Kazakh set.
Language is forced, decoding is greedy, and references and hypotheses pass through the same normalizer everywhere.

| Kazakh WER | params | FLEURS test | KSC test (out of domain) |
|---|---|---|---|
| whisper-tiny / base | 38 / 73 M | 3.104 / 1.157 | — |
| whisper-small | 242 M | 0.770 | 0.863 |
| **whisper-small fine-tuned on 11.8 h Kazakh** | 242 M | **0.238** | 0.469 |
| same, LoRA (3.5 M trainable, 1.4%) | 242 M | 0.238 | 0.483 |
| same + 20% Russian/English rehearsal | 242 M | **0.233** | not measured |
| whisper-large-v3-turbo | 809 M | 0.208 | 0.286 |
| mms-1b-all | 965 M | 0.144 | 0.297 |

For scale: zero-shot whisper-small reaches 0.110 on Russian and 0.071 on English.

## Five findings

1. **Below whisper-small, Kazakh does not work at all.** whisper-tiny returns repetition loops on 60% of Kazakh
   utterances, which is why its WER exceeds 1. The same models are usable on Russian and English, so this is a
   property of the language coverage, not of the architecture.
2. **11.8 h of fine-tuning brings a 242 M model within three WER points of an 809 M one — but only in domain.**
   Fine-tuned whisper-small reaches 0.238 on FLEURS against 0.208 for zero-shot large-v3-turbo. On KSC the ordering reverses (0.469 vs 0.286): the relative
   gain falls from 69% to 46%. A single in-domain number would have overstated the result by a third.
3. **Benchmark contamination is visible in practice.** mms-1b-all looks like the best Kazakh model on FLEURS (0.144)
   but drops to 0.297 on KSC, level with large-v3-turbo; its training data includes the FLEURS train split.
4. **Adapting to Kazakh trades off against Russian, and the update size sets the exchange rate — but rehearsal
   breaks the trade-off.** A LoRA learning-rate sweep moves Kazakh 0.358 → 0.296 → 0.238 while Russian goes
   0.199 → 0.221 → 0.415, so the cost follows the size of the update rather than the tuning method. Adding 20%
   Russian and English data to the fine-tuning set removes about 80% of the forgetting at no measurable cost to
   Kazakh (kk 0.233, ru 0.127 against a 0.110 baseline, en 0.079 against 0.071). Russian suffers far more than
   English throughout: it shares the Cyrillic script and much of the phonology with Kazakh, and the damage is
   phonetic (Russian spelled as heard) rather than a switch of output language.
5. **Local, offline use is practical today.** With CTranslate2 int8 the fine-tuned model is a 253 MB file that keeps
   its accuracy (WER 0.235 on CPU vs 0.238 on GPU) and transcribes one hour of speech in six minutes of CPU time.

## What this suggests for a thesis

- Evaluate low-resource ASR on at least two corpora per language; in-domain gains and contaminated benchmarks are the
  two failure modes this pilot ran into directly.
- Map the adaptation/retention frontier properly: sweep learning rate and rank for both full and parameter-efficient
  tuning, select checkpoints on a multilingual criterion, and test whether interference really tracks language
  similarity across more language pairs.
- Treat the pipeline itself as a measurement instrument: several findings here changed once measurement bugs were
  fixed (30 s truncation, a normalizer deleting bracketed text, `ё`/`е` inconsistencies, number formatting, and
  fine-tuning damaging Whisper's long-form decoding).

All numbers come from full test sets, carry 95% bootstrap confidence intervals, and are reproducible with the
commands in the repository's README; every run stores its per-utterance references and hypotheses. Paired bootstrap
tests confirm the differences quoted above (Kazakh: fine-tuned vs large-v3-turbo −0.030 [−0.052, −0.011]; Russian:
full fine-tuning vs LoRA +0.205 [+0.183, +0.225]) and show that int8 quantization is free within measurement error.
