# Reasoning data inventory

`reasoning-v1` uses pinned and decontaminated sources for the actual production mix. Other available
LUMI sources stay visible here so a later recipe can adopt them without confusing “present on storage”
with “approved for the current run.”

| Source | State | In v1 | Public | LUMI |
|---|---|---:|---:|---:|
| [Multilingual reasoning traces v0.2 pilot](sources/reasoning-traces-multilingual/) | production pilot | all 3,351 16K-eligible rows once | yes | stage into run root |
| [Dolci Think 32B decontaminated](sources/dolci-think-32b-decontaminated/) | production | 20% | yes | stage into run root |
| [Dolci Think 7B decontaminated](sources/dolci-think-7b-decontaminated/) | production | 15% | yes | stage into run root |
| [Nemotron post-training v2](sources/nemotron-post-training-v2/) | production + historical raw | 45% | yes | raw version already shared; decontaminated snapshot staged per run |
| [OpenR1 Math 220K](sources/openr1-math-220k/) | production | 5% | yes | yes |
| [Exact SFT replay](sources/exact-sft-replay/) | production | 15% | model-card lineage | yes |
| [AM DeepSeek R1 distilled](sources/am-deepseek-r1-distilled/) | available alternative | no | upstream needs exact pin | yes |
| [Glaive code assistant v3](sources/glaive-code-assistant-v3/) | available alternative | no | yes | yes |
| [Multilingual Dolci translations](sources/multilingual-dolci-translations/) | evaluation/candidate | no | samples public; full sets access-controlled | partial |
| [Historical reasoning parquet](sources/historical-reasoning-parquet/) | comparison only | no | no standalone artifact | yes |

“Production pilot” means the exact revision, license, acceptance filter, and artifact path are executable,
while retaining the upstream warning that native-speaker and independent solver validation is incomplete.
No dataset artifacts are copied into Git.
