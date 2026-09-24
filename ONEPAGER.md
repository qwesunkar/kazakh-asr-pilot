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
| whisper-large-v3-turbo | 809 M | 0.208 | 0.286 |
| mms-1b-all | 965 M | 0.144 | 0.297 |

For scale: zero-shot whisper-small reaches 0.110 on Russian and 0.071 on English.

## Five findings

1. **Below whisper-small, Kazakh does not work at all.** whisper-tiny returns repetition loops on 60% of Kazakh
   utterances, which is why its WER exceeds 1. The same models are usable on Russian and English, so this is a
   property of the language coverage, not of the architecture.
2. **11.8 h of fine-tuning beats 3.3× more parameters — but only in domain.** Fine-tuned whisper-small reaches 0.238
   on FLEURS against 0.208 for zero-shot large-v3-turbo. On KSC the ordering reverses (0.469 vs 0.286): the relative
   gain falls from 69% to 46%. A single in-domain number would have overstated the result by a third.
3. **Benchmark contamination is visible in practice.** mms-1b-all looks like the best Kazakh model on FLEURS (0.144)
   but drops to 0.297 on KSC, level with large-v3-turbo; its training data includes the FLEURS train split.
4. **Monolingual fine-tuning costs the other languages, and LoRA did not prevent it.** Full fine-tuning moves Russian
   WER 0.110 → 0.210 and English 0.071 → 0.106. LoRA matches full fine-tuning on Kazakh (0.238) with 1.4% of the
   parameters and half the VRAM, but Russian ends at 0.415 — worse than full fine-tuning. The damage is phonetic
   (Russian spelled as heard) rather than a switch of output language. The two recipes differ in learning rate by
   100×, so isolating the cause is a concrete first experiment.
5. **Local, offline use is practical today.** With CTranslate2 int8 the fine-tuned model is a 253 MB file that keeps
   its accuracy (WER 0.235 on CPU vs 0.238 on GPU) and transcribes one hour of speech in six minutes of CPU time.

## What this suggests for a thesis

- Evaluate low-resource ASR on at least two corpora per language; in-domain gains and contaminated benchmarks are the
  two failure modes this pilot ran into directly.
- Pin down why parameter-efficient tuning degraded a related language here (Russian shares the Cyrillic script and
  much of the phonology with Kazakh), separating learning rate, rank and target modules.
- Treat the pipeline itself as a measurement instrument: several findings here changed once measurement bugs were
  fixed (30 s truncation, a normalizer deleting bracketed text, `ё`/`е` inconsistencies, number formatting, and
  fine-tuning damaging Whisper's long-form decoding).

All numbers come from full test sets and are reproducible with the commands in the repository's README; every run
stores its per-utterance references and hypotheses.
