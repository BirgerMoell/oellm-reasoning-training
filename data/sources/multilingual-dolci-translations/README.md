# Multilingual Dolci translations

| Field | Value |
|---|---|
| State | `candidate/evaluation`, not in production v1 |
| Role | broaden explicit reasoning beyond English and the Nemotron language set |
| Public sample | [`ezosa/Dolci-Think-SFT-7B-translations`](https://huggingface.co/datasets/ezosa/Dolci-Think-SFT-7B-translations) |
| Languages reviewed | Czech, German, Finnish, French, Italian, Spanish, Swedish |
| Full 32B translations | 500K-row German, French, Spanish and other subsets exist, but access/metadata must be checked per repository |
| License | inherited/composite; resolve before training |

The sample translations are useful to choose a translation model and trace policy, not large enough for
production training. Project discussions also record a real language-switching risk when reasoning is
English-only, so v1 allocates 20% to native multilingual Nemotron traces and measures output-language
adherence. A future v2 should promote reviewed translations only after language-ID, mathematical
equivalence, answer verification, decontamination, and a pinned public revision are available.
