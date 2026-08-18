# Artifact and lineage contract

Git stores recipes, code, documentation, and small manifests. LUMI scratch stores model/data artifacts.
Hugging Face stores immutable public releases. A friendly name is never sufficient to identify a run.

## LUMI layout

```text
$OELLM_RUN_ROOT/
  cache/huggingface/
  models/oellm-9b-256k-sft/
  raw/
    datasets/<owner--dataset>/
  data/reasoning-v1/
    train.parquet
    manifest.json
    dedup.sqlite3
  data/reasoning-sanity/       # disposable sampled integration artifact
  checkpoints/
    reasoning-v1-smoke/
    reasoning-v1/checkpoint-{250,500,750,1000,1250,1500,1750,2000}/
  eval/
    baseline-08359ad/
    reasoning-v1-step-0500/
  runs/<run-id>/
    run.yaml
    resolved-data-config.yaml
    resolved-train-config.yaml
    environment.txt
    slurm.txt
    metrics.jsonl
  releases/<model-name>/
```

## Run record

`run.yaml` must contain:

- repository URL and Git SHA, dirty/clean state;
- parent model repository, revision, local path, and config SHA;
- data recipe name, recipe SHA, materialized-manifest SHA, Parquet SHA;
- resolved training configuration and template SHA;
- container path/hash and overlay/package versions;
- Slurm account, partition, job IDs, nodes, GCDs, elapsed time, and GPU-hours;
- resume/warm-start history;
- checkpoint paths and hashes;
- evaluation directories and final decision.

## Data manifest

The builder writes:

- input repository IDs/revisions or absolute local paths;
- resolved input files and sizes;
- raw, invalid, duplicate, overlength, and selected row counts;
- selection strategy (`all_once` or `token_weighted`) and per-language counts;
- selected rendered tokens and achieved token share per slice;
- tokenizer/model revision and template SHA;
- seed, builder Git SHA when available, build host/time;
- output row count, token count, and SHA-256.

Copy the small manifest into the eventual model repository. It lets a reader locate large data without
placing the data in Git.

## Model lineage

```text
openeurollm/oellm-9b-256k-theta64m-prelude
  -> birgermoell/oellm-9b-256k-sft@08359ad
      -> reasoning-v1 step checkpoints
          -> accepted reasoning checkpoint (only after gates)
              -> later RLVR / preference / tool stages in separate repositories
```

The starting SFT checkpoint itself used 1,082,196 general conversations plus the first 250,000 records of
a concatenated 1,526,602-row historical reasoning parquet. This repository keeps that history visible but
does not reuse concatenation order as a mixing mechanism.
