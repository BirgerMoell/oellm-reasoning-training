# LUMI run 22211014: Anneal-300B reasoning timeout

## Outcome

The first production attempt was **not completed and produced no checkpoint**. Slurm killed the job at
the five-hour wall limit after the trainer completed step 15 and stopped completing further updates.
Do not represent this attempt as a trained model.

| Field | Value |
|---|---|
| Slurm job | `22211014` |
| Started | 2026-09-22 08:38:43 |
| Ended | 2026-09-22 13:38:53 |
| State | `TIMEOUT` |
| Allocation | 8 LUMI-G nodes / 64 MI250X GCDs |
| Repository | `279e4da1490c7cbd2911d92274ed9ea9058d67a5` |
| Parent revision | `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31` |
| Data manifest SHA-256 | `920b50ce5a72fa925e28383a037178da418cff4e64b2a89d8da5839a14bad9b1` |
| Training parquet SHA-256 | `b8f23c0e709e0a0d72b0926ff09a13942229a0c17d690feb171fa70d6c9d4499` |
| Last completed update | 15 of 500 |
| Checkpoints | none; the first scheduled save was step 100 |

## Evidence

- Data build job `22211011`, ten-step sanity job `22211012`, and exhaustive parent-weight scan job
  `22211013` all completed. The scan found all 9,101,947,904 model values finite.
- Updates 1–15 completed at roughly 17.8 seconds per update. No rank printed completion of update 16.
- The step-10 window was finite and plausible: loss `0.8346`, gradient norm `0.4778`, learning rate
  `9e-7`, token accuracy `0.7722`, and 10,469,376 processed tokens.
- Sampled GPUs remained at 100% activity after progress stopped. That can occur while a ROCm kernel or
  RCCL collective spins and is not evidence that useful training was continuing.
- There was no Python traceback. The launcher had a five-hour DDP timeout and no shorter process-group
  watchdog or collective flight recorder, so Slurm supplied the only termination.
- The run used Liger Kernel 0.8.1 fused linear cross-entropy with Triton 3.2.0. Its runtime repeatedly
  warned that the Triton version was below the supported recommendation.

## Diagnosis and resolution gate

The evidence is most consistent with a fused GPU-kernel or downstream RCCL collective stall. It does
not identify the unique offending kernel, so this remains a high-confidence operational diagnosis rather
than proof of a Liger defect.

The recovery replaces only the loss implementation with TRL 1.4.0 `chunked_nll`, while preserving the
parent, immutable training parquet, exact ChatML assistant mask, optimizer, scheduler, BF16 precision,
16K packing, and 64-GCD topology. It also adds a 10-minute process-group watchdog, a 15-minute DDP
timeout, collective diagnostics, per-rank RCCL logs, and per-node step heartbeats.

Before production is resubmitted, the production-identical recovery gate must:

1. complete 30 updates, crossing the previous step-15 boundary;
2. save full trainer-state checkpoints at steps 10, 20, and 30;
3. keep loss and gradient norm finite; and
4. pass an exhaustive scan of every value in `checkpoint-30`.
