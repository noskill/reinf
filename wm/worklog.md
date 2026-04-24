# World Model Experiment Worklog

Date: 2026-03-24

## Summary (from log files)

| Log file | Status | Config snapshot | Final logged metrics |
|---|---|---|---|
| `wm_tssm_prior_rollout_130e.log` | Complete (`130` epochs, checkpoint saved) | `model=tssm`, `prior_next_w=0.0`, `prior_roll_w=0.5`, `bptt_horizon=0` | Epoch `130`: train loss `16.6621`; val loss `26.0159`; val `lr_acc=0.478`; val `loc_x_acc=0.146`, `loc_y_acc=0.119`; val `kl_dyn=11.9638`, `kl_rep=0.5982`; val `prior_roll=4.6982` |
| `wm_tssm_prior_next_130e.log` | **Incomplete** (stops at `starting epoch 001`) | `model=tssm`, `prior_roll_w=0.0`, `z_only_w=0.0`, `h_only_w=0.0`, `bptt_horizon=0` | No epoch metrics; no checkpoint save line |
| `wm_tssm_both_130e.log` | Complete (`130` epochs, checkpoint saved) | `model=tssm`, `prior_next_w=2.0`, `prior_roll_w=0.5`, `bptt_horizon=0` | Epoch `130`: train loss `26.4078`; val loss `30.6627`; val `lr_acc=0.494`; val `loc_x_acc=0.137`, `loc_y_acc=0.120`; val `kl_dyn=8.3565`, `kl_rep=0.4178`; val `prior_next=4.1951`, `prior_roll=4.7199` |
| `wm_tssm_baseline_300e.log` | Complete (`300` epochs, checkpoint saved) | (no `run config` line present in log; older format) | Epoch `300`: train loss `12.3088`; val loss `13.6274`; val `lr_acc=0.293`; val `loc_x_acc=0.139`, `loc_y_acc=0.118`; val `kl_dyn=2.2682`, `kl_rep=0.2268`; val `prior_next=4.4411` |
| `wm_tssm_prior_rollout_130e_probes.log` | Complete (`30` epochs, checkpoint saved) | `model=tssm`, `prior_roll_w=0.0`, `z_only_w=1.0`, `h_only_w=1.0`, `recon_heads_only=True` | Epoch `030`: train loss `11.2653`; val loss `11.7248`; val `lr_acc=0.471`; val `loc_x_acc=0.145`, `loc_y_acc=0.121`; val `z_only=3.8702`, `h_only=4.2924`; val `prior_roll=0.0000` |

## Notes

- `wm_tssm_prior_next_130e.log` appears interrupted early and is not usable for final comparison.
- `wm_tssm_baseline_300e.log` appears to come from an older logging format (no `run config` line), so direct comparison with newer logs should be treated carefully.
- The probes run (`wm_tssm_prior_rollout_130e_probes.log`) is head-only diagnostic training (`recon_heads_only=True`), not full backbone world-model training.

## Training Regime Differences

The key difference across runs is which prior-model losses were optimized during training.

| Regime label | Main idea | Prior-next loss | Prior-rollout loss | Probe-only losses (`z_only`, `h_only`) | Backbone trainable |
|---|---|---|---|---|---|
| `baseline` | Teacher-forced / non-rollout baseline | On in older logs (see `prior_next` metric in `wm_tssm_baseline_300e.log`) | Off | Off | Yes |
| `prior_next` | Focus on one-step prior prediction | Intended On (by experiment name; log file is incomplete) | Off (`prior_roll_w=0.0` in log header) | Off | Yes |
| `prior_rollout` | Multi-step open-loop prior rollout | Off (`prior_next_w=0.0` in run config) | On (`prior_roll_w=0.5`) | Off | Yes |
| `both` | Combine one-step and rollout prior constraints | On (`prior_next_w=2.0`) | On (`prior_roll_w=0.5`) | Off | Yes |
| `prior_rollout_probes` | Diagnostic head fitting on frozen WM | Off (`prior_roll_w=0.0`) | Off | On (`z_only_w=1.0`, `h_only_w=1.0`) | **No** (heads only) |

Interpretation shorthand:
- `prior_next` pushes local one-step latent consistency.
- `prior_rollout` pushes long-horizon open-loop latent consistency.
- `both` enforces both local and long-horizon consistency.
- `probes` does not improve dynamics; it measures what information is decodable from learned latent states.

## Reverse-Path Test Results (checkpoint benchmark)

Settings used for all rows:
- `wm_reverse_path_test.py`
- `sensor_mode=categorical`
- `sensor_head=zh`
- `n_step=10`
- `max_rollout_steps=10`
- `max_val_episodes=50`
- `device=cpu`

| Checkpoint | episodes | steps | lr_acc | loc_acc | after_turn_step_lr_acc | final_start_lr_acc |
|---|---:|---:|---:|---:|---:|---:|
| `outputs/wm_tssm_baseline_130e.pt` | 50 | 500 | 0.1260 | 0.0020 | 0.1000 | 0.1200 |
| `outputs/wm_tssm_prior_next_130e.pt` | 50 | 500 | 0.1500 | 0.0020 | 0.1000 | 0.1000 |
| `outputs/wm_tssm_prior_rollout_130e.pt` | 50 | 500 | 0.2100 | 0.0280 | 0.1600 | 0.4200 |
| `outputs/wm_tssm_both_130e.pt` | 50 | 500 | 0.2280 | 0.0100 | 0.2400 | 0.4000 |
| `outputs/wm_tssm_prior_rollout_130e_probes.pt` | 50 | 500 | 0.1940 | 0.0280 | 0.2000 | 0.3400 |

### Reverse test notes

- The strongest `zh` reverse-path scores in this run are from `both_130e` and `prior_rollout_130e`.
- `baseline_300e` log exists, but `outputs/wm_tssm_baseline_300e.pt` was not present in `outputs/`, so reverse-path benchmark used `outputs/wm_tssm_baseline_130e.pt`.

## Next Plan (agreed)

1. Non-latent baseline:
   - Train `model_type=transformer` as deterministic next-observation predictor (`h_t -> \hat{o}_{t+1}`).
   - Script: `wm_train_transformer_next_130e.sh`.
2. Better base Dreamer/Twister-style training:
   - Tune core hyperparameters (LR, KL weights, temperature, free nats) while keeping architecture and base loss regime fixed (`prior_rollout_weight=0.0`).
   - Script: `wm_train_tssm_tuned_130e.sh`.
3. Limited forward rollout training:
   - Use open-loop prior rollout loss only for first `K` rollout steps (new `--prior-rollout-steps` argument; `0=full`).
   - Script for `K=10`: `wm_train_prior_rollout_steps10_130e.sh`.

Implementation notes:
- Transformer path is currently categorical-sensor only to keep training/evaluation simple.
- New arg in training: `--prior-rollout-steps` (applies to RSSM/TSSM prior-rollout loss window).

## Script Inventory

| Script | Regime | Epochs | Log | Checkpoint | Reverse-path | Latest reverse (`lr_acc/loc_acc`) |
|---|---|---:|---|---|---|---|
| `wm_train_base.sh` | TSSM baseline (soft targets, no rollout) | 130 | `wm_tssm_baseline_130e.log` | `outputs/wm_tssm_baseline_130e.pt` | Yes (`n=10`) | `0.126/0.002` |
| `wm_train_prior_next.sh` | TSSM prior-next style (current script uses no rollout aux) | 130 | `wm_tssm_prior_next_130e.log` | `outputs/wm_tssm_prior_next_130e.pt` | Yes (`n=10`) | `0.150/0.002` |
| `wm_train_prior_rollout.sh` | TSSM prior-rollout (full-sequence open-loop aux) | 130 | `wm_tssm_prior_rollout_130e.log` | `outputs/wm_tssm_prior_rollout_130e.pt` | Yes (`n=10`) | `0.210/0.028` |
| `wm_train_prior_rollout_probes.sh` | TSSM probe/head-only training on frozen backbone | 30 | `wm_tssm_prior_rollout_130e_probes.log` | `outputs/wm_tssm_prior_rollout_130e_probes.pt` | Yes (`n=10`) | `0.194/0.028` |
| `wm_train_prior_rollout_steps10_130e.sh` | TSSM prior-rollout limited to 10 steps | 130 | `wm_tssm_prior_rollout_steps10_130e.log` | `outputs/wm_tssm_prior_rollout_steps10_130e.pt` | Yes (`n=10`) | `0.160/0.012` |
| `wm_train_tssm_tuned_130e.sh` | TSSM tuned baseline hyperparameters | 130 | `wm_tssm_tuned_130e.log` | `outputs/wm_tssm_tuned_130e.pt` | Yes (`n=10`) | `0.082/0.002` |
| `wm_train_transformer_next_130e.sh` | Non-latent transformer next-sensor baseline (full context) | 300 | `wm_transformer_next_300e.log` | `outputs/wm_transformer_next_300e.pt` | Yes (`n=5`) | `0.048/0.020` |
| `wm_train_transformer_next_ctx5_300e.sh` | Transformer next-sensor ablation with `context_len=5` | 300 | `wm_transformer_next_ctx5_300e.log` | `outputs/wm_transformer_next_ctx5_300e.pt` | Yes (`n=5`) | `0.244/0.032` |
| `wm_train_rnn_next_300e.sh` | Non-latent RNN (GRU) next-sensor baseline | 300 | `wm_rnn_next_300e.log` | `outputs/wm_rnn_next_300e.pt` | Yes (`n=5`) | `0.136/0.020` |
| `wm_train_rssm_next_300e.sh` | RSSM-discrete next-sensor baseline (no rollout/z/h aux) | 300 | `wm_rssm_next_300e.log` | `outputs/wm_rssm_next_300e.pt` | No | `-` |

Notes:
- `wm_train_transformer_next_130e.sh` is a legacy script name; it currently runs `300` epochs and writes `*_300e` outputs.

## Update (2026-03-25)

### Recent Results Summary

| Run / Log | Final train lr_acc | Final val lr_acc | Final val loc_x_rmse | Final val loc_y_rmse | Reverse-path tested? | Notes |
|---|---:|---:|---:|---:|---|---|
| `wm_transformer_next_300e.log` (regularized full context) | 0.720 | 0.519 | 4.134 | 4.534 | Yes (`n=5`) | Best recent one-step val LR among non-latent baselines |
| `wm_rnn_next_300e.log` | 0.575 | 0.557 | 4.086 | 4.484 | Yes (`n=5`) | Best one-step val LR so far; location probe still weak |
| `wm_transformer_next_ctx5_300e.log` (`context_len=5`) | 0.535 | 0.436 | 4.140 | 4.466 | Yes (`n=5`) | Short-context ablation drops one-step val LR vs full context, but improves short reverse-path metrics |
| `wm_tssm_tuned_130e.log` | 0.551 | 0.446 | 4.067 | 4.469 | Yes (`n=10`) | Tuned TSSM baseline, no prior-rollout aux |
| `wm_tssm_prior_rollout_steps10_130e.log` | 0.559 | 0.439 | 4.075 | 4.464 | Yes (`n=10`) | Limited prior-rollout (`prior_rollout_steps=10`) |

Aux baseline:
- Repeat baseline (`--baseline repeat`) on same split gives `val lr_acc=0.114`, much lower than learned models.
- Reverse-path results are reported in separate sections and may use different horizons (`n=5` vs `n=10`).

### Reverse-Path n=5 (new)

Settings:
- `wm_reverse_path_test.py --sensor-mode categorical --sensor-head zh --n-step 5 --max-rollout-steps 5 --max-val-episodes 50 --device cpu`

| Checkpoint | lr_acc | loc_acc | after_turn_step_lr_acc | final_start_lr_acc |
|---|---:|---:|---:|---:|
| `outputs/wm_transformer_next_300e.pt` | 0.0480 | 0.0200 | 0.1000 | 0.0200 |
| `outputs/wm_rnn_next_300e.pt` | 0.1360 | 0.0200 | 0.1000 | 0.1000 |
| `outputs/wm_transformer_next_ctx5_300e.pt` | 0.2440 | 0.0320 | 0.1800 | 0.3000 |
| `outputs/wm_tssm_tuned_130e.pt` | 0.1680 | 0.0000 | 0.1600 | 0.3200 |
| `outputs/wm_tssm_prior_rollout_steps10_130e.pt` | 0.1280 | 0.0040 | 0.1200 | 0.2200 |

Takeaway:
- One-step val LR can be moderate/good while reverse open-loop performance remains poor.
- `context_len=5` improved short-horizon (`n=5`) reverse metrics vs other non-latent baselines, despite lower one-step val LR.

### Residual RNN Follow-up (2026-03-25)

Status:
- Residual RNN (`h_t = h_{t-1} + g_t`) is now a valid option to explore further.

Reverse-path `n=5` quick comparison:
- `outputs/wm_rnn_next_300e.pt` (GRU): `lr_acc=0.1360`, `loc_acc=0.0160`, `final_start_lr_acc=0.0600`
- `outputs/wm_rnn_residual_next_300e.pt` (Residual): `lr_acc=0.3400`, `loc_acc=0.0200`, `final_start_lr_acc=0.5200`

Interpretation:
- Residual transition substantially improves reverse-path LR prediction vs GRU in this setup.
- Location accuracy remains low for both, so this should be treated as a promising but incomplete direction.

### Objective Consistency Check (2026-03-25)

- Non-latent next-sensor baselines (`transformer`, `rnn`, `rnn-residual`) were already consistent: `--turn-weight 0.0 --step-weight 0.0`.
- Older TSSM scripts (`base/prior_next/prior_rollout/prior_rollout_steps10/tuned`) did not explicitly zero turn/step, so they used default action-loss weights.
- Training code is now aligned to probe-style auxiliaries: location/heading and turn/step heads use detached state inputs (do not update backbone dynamics).
- For strict cross-regime comparison, rerun key TSSM checkpoints under the new consistent objective settings.

### New Runs Started (2026-03-25)

Started long runs:
- `wm_train_tssm_tuned_130e.sh` -> `outputs/wm_tssm_tuned_600e.pt` (`wm_tssm_tuned_600e.log`)
- `wm_train_rssm_next_300e.sh` -> `outputs/wm_rssm_next_600e.pt` (`wm_rssm_next_600e.log`)
- `wm_train_rssm_residual_next_600e.sh` -> `outputs/wm_rssm_residual_next_600e.pt` (`wm_rssm_residual_next_600e.log`)

Next diagnostic requirement:
- We need controlled reconstruction/probe evaluation from `z_t`, `h_t`, and `s_t=[z_t,h_t]` to verify whether latent state carries useful information.
- This should be compared under identical probe capacity/training budget and the same evaluation splits/horizons.



i think perhaps rnn is undertrained and transformer simply don't expose these info in cls token; otherwise lr sensor is impossible to predict without location, esp location would be usefull on later stages of
  trajectory

## Update (2026-03-26)

### New Results

1. RSSM baseline (`wm_train_rssm_next_300e.sh`, now 600 epochs) completed:
- Checkpoint: `outputs/wm_rssm_next_600e.pt`
- Final log line (`wm_rssm_next_600e.log`):
  - train: `lr_acc=0.412`, `kl_dyn=2.8145`, `kl_rep=0.0938`
  - val: `lr_acc=0.484`, `loc_x_rmse=3.842`, `loc_y_rmse=4.408`, `kl_dyn=3.9731`, `kl_rep=0.1324`

2. Reverse-path test for `outputs/wm_rssm_next_600e.pt`:
- `n=5`, `max_rollout_steps=5`: `lr_acc=0.3920`, `loc_acc=0.1200`, `final_start_lr_acc=0.4000`
- `n=10`, `max_rollout_steps=10`: `lr_acc=0.3420`, `loc_acc=0.0180`, `final_start_lr_acc=0.3200`

3. Speed benchmark (`wm_bench_rssm_tssm_speed.py`) on GPU:
- Batch `8`, seq `64`, mode `both`: TSSM is ~`8.03x` slower (eval), ~`7.86x` slower (train) than RSSM.
- Batch `8`, seq `128`, mode `train`: TSSM is ~`7.69x` slower than RSSM.

4. modifications with skip-connection are much worse, requires additional tuning.

4. TSSM tuned long run status:
- `wm_tssm_tuned_600e.log` still running; latest observed around epoch `352`.
- Pattern remains high train/val gap and large val `kl_dyn`, suggesting current no-rollout regime is likely near plateau for open-loop goals.

### Theory / Direction Update

- Current maze dataset has partial observability/aliasing; exact multi-step observation prediction is often ambiguous.
- Multi-step rollout loss likely acts partly as regularization/consistency pressure, not strict deterministic future prediction.
- Twister-style embedding rollout with contrastive objective (AC-CPC-like) remains a higher-priority direction.
- HMM-style smoothing idea (trajectory-level latent smoothing / forward-backward style inference) is also promising, but lower priority than AC-CPC-style objective for now.
- other option to try is cls token in transformer vs current token output

### Implementation Update (2026-03-26, contrastive heads)

- Added optional next-step contrastive objective (`InfoNCE`) to all model paths:
  - Transformer / RNN: embedding head on `h_t`
  - RSSM / TSSM: embedding head on `s_t=[h_t,z_t]`
- Contrastive positives: `(e_t, e_{t+1})` from the same batch row.
- Negatives: valid `e_{t+1}` from other rows in the same batch.
- Simplifying assumption now used in code: each batch row is from a different episode.
- Added TODO notes for future improvements:
  - negatives from virtual rollouts
  - hard negatives from same episode but temporally distant timesteps
- Safety/default behavior:
  - contrastive is **disabled by default** now (`--contrastive-weight 0.0`, `--contrastive-dim 0`)
  - enable explicitly in runs when desired (to avoid silently changing legacy baselines).

## Update (2026-03-30)

- RNN residual mode implementation detail:
  - current code uses a fast proxy residual update `h = cumsum(h_raw * scale)` after GRU forward.
  - this is not a true per-step residual recurrent cell update (`h_t = h_{t-1} + g(x_t, h_{t-1})`).
- Transformer baseline sanity:
  - overfit check confirms the baseline can memorize under low-regularization settings.
- Sensor smoothing note:
  - large `--sensor-sigma` hurts exact LR-bin fit in this setup and can make training look worse on `lr_acc`.
  - for strict overfit diagnostics, prefer `--sensor-sigma 0.0`.

## Update (2026-03-31)

### GRU OFAT Sweep (train/val summary)

Settings:
- `model_type=rnn`, `rnn_transition=gru`, sensor-only objective (`loc/head/turn/step=0`), categorical sensors.
- OFAT phases in `wm_sweep_rnn_gru_tune.sh`: norm, lr, wd, sigma, capacity.

Best per phase (final logged epoch):

| Phase | Best run | Train `lr_acc` | Val `lr_acc` | Val `lr_rmse` |
|---|---|---:|---:|---:|
| norm | `wm_rnn_gru_ofat_norm_nlayernorm_lr3em4_wd0p0_ss0p0_h128_l2_200e.log` | 0.592 | 0.555 | 1.099 |
| lr | `wm_rnn_gru_ofat_lr_nrmsnorm_lr3em4_wd0p0_ss0p0_h128_l2_200e.log` | 0.589 | 0.552 | 1.099 |
| wd | `wm_rnn_gru_ofat_wd_nrmsnorm_lr3em4_wd1em2_ss0p0_h128_l2_200e.log` | 0.588 | 0.554 | 1.097 |
| sigma | `wm_rnn_gru_ofat_sigma_nrmsnorm_lr3em4_wd0p0_ss1p0_h128_l2_200e.log` | 0.580 | 0.560 | 1.129 |
| capacity* | `wm_rnn_gru_ofat_capacity_nrmsnorm_lr3em4_wd0p0_ss0p0_h128_l2_200e.log` | 0.589 | 0.552 | 1.099 |

\*In the first OFAT capacity phase, several width/depth runs failed due fixed `head_dim=32` constraint (`hidden_size != heads * head_dim`). Sweep scripts were patched to set `head_dim = hidden_size / 4`.

Notes:
- `lr=1e-3` strongly overfit (`train lr_acc=0.960`, `val lr_acc=0.456`).
- Smoother sensor targets improved val `lr_acc` in this metric set (`sigma=0.5/1.0`), with `sigma=1.0` trading worse `val lr_rmse`.

### GRU Capacity Re-run (fixed head_dim)

Settings:
- `lr=3e-4`, `wd=0.0`, `sensor_sigma=0.0`, `rnn_state_norm=rmsnorm`, `epochs=200`.
- Logs: `wm_rnn_gru_capacity_*_200e.log`.

| Hidden/Layers | Train `lr_acc` | Val `lr_acc` | Val `lr_rmse` |
|---|---:|---:|---:|
| `128 / 2` | 0.589 | 0.552 | 1.099 |
| `128 / 6` | 0.818 | 0.504 | 1.321 |
| `136 / 5` | 0.860 | 0.510 | 1.321 |
| `152 / 4` | 0.886 | 0.495 | 1.389 |
| `176 / 3` | 0.889 | 0.476 | 1.350 |
| `216 / 2` | 0.748 | 0.506 | 1.232 |

Interpretation:
- Increasing capacity significantly improves train fit but hurts validation in this current regime.
- Best val result in this re-run remains `128/2`.

### Additional Targeted Runs

| Run | Train `lr_acc` | Val `lr_acc` | Val `lr_rmse` |
|---|---:|---:|---:|
| `wm_rnn_gru_h128_l3_200e.log` | 0.639 | 0.551 | 1.124 |
| `wm_rnn_gru_h256_l3_200e.log` | 0.974 | 0.478 | 1.420 |

Takeaway:
- `h128,l3` is close to baseline val accuracy but not better.
- `h256,l3` heavily overfits with clear train/val gap.



next plan:

pretrain transformer/rnn; use smaller number of epochs for word modelling losses

## Update (2026-03-31, Pretraining Baselines)

### RNN Completed

- Transformer pretraining completed.
- Script: `wm_pretrain_transformer_h176l3.sh`
- Log: `wm_transformer_pretrain_h176l3.log`
- Checkpoint: `outputs/wm_transformer_pretrain_h176l3.pt`
- Final (epoch 300): train `lr_acc=0.968`, val `lr_acc=0.527`, val `lr_rmse=1.305`.

### Completed

- RNN pretraining run A (RMSNorm variant) completed.
  - Script: `wm_pretrain_rnn_h176l3.sh`
  - Log: `wm_rnn_pretrain_h176l3.log`
  - Checkpoint: `outputs/wm_rnn_pretrain_h176l3.pt`
  - Config note: `rnn_state_norm=rmsnorm`
  - Final (epoch 300): train `lr_acc=0.806`, val `lr_acc=0.506`, val `lr_rmse=1.260`.

- RNN pretraining run B (no norm variant) completed.
  - Script: `wm_pretrain_rnn_h176l3.sh` with `RNN_NORM=none` variant naming.
  - Log: `wm_rnn_pretrain_no_norm_h176l3.log`
  - Checkpoint: `outputs/wm_rnn_pretrain_no_norm_h176l3.pt`
  - Config note: `rnn_state_norm=none`
  - Final (epoch 300): train `lr_acc=0.645`, val `lr_acc=0.538`, val `lr_rmse=1.125`.

### Twister Direction Note

- We need to move towards a Twister-like model.
- Add continuous embedding heads:
  - baselines: `h_t -> e_t`
  - world models: use a predictor over prior features and future actions.
- Corrected formulation from code reading:
  - `f(prior_feat_t, a_t, ..., a_{t+k-1})`
  - where `prior_feat_t = get_feat(prior_t)` and includes prior `\hat{z}_t` plus deterministic state `h_t`.
- Contrastive target branch:
  - in our implementation target is explicitly `z_t -> e_t` projection (posterior stochastic latent from augmented observations).
- Contrastive optimization detail:
  - for the contrastive term, gradients should flow through both branches.
- In the Twister paper they mention virtual rollout with 10 future actions; code behavior should be interpreted via the `prior_feat_t` + action-sequence function above.
- We need to investigate this discrepancy in detail.
- For now, implement a similar setup in our codebase and test it.

## Update (2026-04-02, AC-CPC Implementation + Run Started)

### Implemented (code)

- Added Twister-style action-conditioned contrastive branches for world models:
  - `f_t([prior_feat_t, a_t, ..., a_{t+h-1}]) -> e_{t+h}` for horizons `h=1..K`.
  - Implemented in RSSM/TSSM paths via `contrastive_pred_emb_steps`.
- Kept temperature scaling + feature normalization in InfoNCE.
- Updated horizon weighting to Twister-style normalized exponential:
  - branch index `t` weight = `lambda^t / sum_j lambda^j`.
- Updated negatives to include same-row valid keys (full-matrix style), excluding only the matched positive.
- Added optional sampled negatives for speed:
  - new arg: `--contrastive-negatives` (`0` = full matrix, `>0` = sample N negatives per anchor).
- Added contrastive accuracy metric similar to Twister:
  - `cpc_acc` = fraction of rows where positive index is top-1 logit.
- Made both projection branches nonlinear (2-layer MLP):
  - predictor branch (`prior/action -> e`) now MLP per horizon.
  - target branch (`z_t -> e_t`) now MLP.

### New/Updated Training Interface

- New args in `wm_train.py`:
  - `--contrastive-steps`
  - `--contrastive-horizon-discount`
  - `--contrastive-negatives`
- Epoch logs now include `cpc_acc`.

### Script Added/Updated

- Script: `wm_train_rssm_from_rnn_accpc.sh`
- Default warm start:
  - `outputs/wm_rnn_pretrain_h176l3.pt`
- Current defaults:
  - `EPOCHS=600`
  - `CPC_WEIGHT=0.2`
  - `CPC_DIM=128`
  - `CPC_STEPS=10`
  - `CPC_DISCOUNT=0.75`
  - `CPC_NEGATIVES=128`
  - `CONTEXT_LEN=32` (set `CONTEXT_LEN=0` for full-episode context)

### Current Progress

- Started new RSSM run from pretrained RNN checkpoint with AC-CPC configuration.
- Smoke checks passed for:
  - multi-step CPC
  - weighted CPC
  - sampled negatives mode
  - `cpc_acc` logging

## Update (2026-04-03, RSSM from RNN Pretrain + AC-CPC)

Checkpoint:
- `outputs/wm_rssm_from_rnn_pretrain_h176l3_accpc_600e.pt`

Final training log (epoch 600):
- train: `loss=9.8778`, `lr_acc=0.460`, `loc_x_rmse=3.897`, `loc_y_rmse=3.927`, `kl_dyn=2.7275`, `kl_rep=0.0909`, `cpc=2.9663`, `cpc_acc=0.252`
- val: `loss=11.1011`, `lr_acc=0.484`, `loc_x_rmse=3.847`, `loc_y_rmse=4.268`, `kl_dyn=3.4961`, `kl_rep=0.1165`, `cpc=3.2241`, `cpc_acc=0.200`

Reverse-path evaluation (categorical, `sensor_head=zh`, `max_val_episodes=50`, cpu):

| Checkpoint | n-step | steps | lr_acc | loc_acc | after_turn_step_lr_acc | final_start_lr_acc |
|---|---:|---:|---:|---:|---:|---:|
| `outputs/wm_rssm_from_rnn_pretrain_h176l3_accpc_600e.pt` | 5 | 250 | 0.4120 | 0.0840 | 0.2000 | 0.4400 |
| `outputs/wm_rssm_from_rnn_pretrain_h176l3_accpc_600e.pt` | 10 | 500 | 0.2800 | 0.0160 | 0.2200 | 0.2400 |

Notes:
- This run was trained with full context (`CONTEXT_LEN=0`).
- AC-CPC objective is clearly active (`cpc_acc` well above random), but one-step validation `lr_acc` remains close to prior RSSM baseline range.


from now we experiment with:
wm_pretrain_rnn_h176l3.sh
wm_pretrain_transformer_h128l3.sh
then train full world model:

1. Pretrain baseline backbone (h features) with sensor + AC-CPC.
2. Initialize world model from that checkpoint (partial load as available).
3. Train RSSM/TSSM with CPC enabled (contrastive_weight > 0) so:
    - predictor branch (prior_feat) learns,
    - target branch (z -> e) learns,
    - z participates in representation shaping.
4. Optionally ramp CPC weight down later, but not to zero too early.

## Update (2026-04-06, Joint WM+Policy from Scratch Plan)
goal - train policy with world model

Plan (intrinsic-only, staged):
1. Build a minimal trainable maze wrapper from existing `tester.py` dynamics.
2. Phase A (no latent `z`): train baseline (`model-type=rnn` or `transformer`) + policy jointly from scratch.
3. Use intrinsic reward only for policy optimization: `r_t = beta * MI_proxy_t` (no environment reward in this phase).
4. Keep probe losses as diagnostics only (they do not shape backbone); backbone shaping comes from sensor objective + AC-CPC.
5. Phase B (latent on): switch to `rssm-discrete`/`tssm` and compare emergent behavior against Phase A under the same intrinsic-reward setup.
6. Track: coverage, loop/cycle tendency, episode length, `cpc_acc`, and transfer to goal-reaching when extrinsic reward is re-enabled later.

## Update (2026-04-08, Baseline Cache API Unification)

Completed:
- Removed hardcoded maze WM model construction path and wired `create_maze_world_model` through shared `create_model(...)` args.
- Added shared parser helpers for model creation args so `wm_train.py` and `wm_maze_train.py` use consistent model configuration plumbing.
- Added internal cache/state handling to baseline predictors:
  - `UnifiedPredictor`: transformer KV cache lifecycle (`init_cache/reset_cache/clear_cache`) + `episode_start` step path.
  - `RNNPredictor`: same API surface, backed by recurrent hidden-state + previous-action state.

Next direction:
1. Switch policy rollout wrappers to call predictor `forward(..., episode_start=...)` directly (remove duplicate rollout-state bookkeeping where possible).
2. Keep sequence training path unchanged; use cache/state path only for online episode collection.
3. Then move to WM/baseline cache semantics cleanup (single consistent online API), before enabling latent-state (`z`) variants.

## Update (2026-04-09, Refactor Plan Execution)

Completed:
- Predictor contract is now explicit and validated in one shared helper (`UnifiedPredictor._validate_obs_contract`):
  - train: `sensor`, `actions`, optional `key_padding_mask`
  - rollout: `sensor`, `actions`, `prev_actions`, `episode_start`
- `WMActionHeadPolicy` now uses `wm_model.forward(..., episode_start=...)` directly and no longer keeps duplicate hidden/KV state.
- AC-CPC intrinsic reward computation moved to `wm_intrinsic.py` (`compute_step_accpc_reward`), keeping env reward passthrough unchanged.
- Removed transitional alias `JointWMAgent`.
- Shared maze model arg defaults moved to `agent_utils_wm.py` (`MAZE_WM_MODEL_DEFAULTS`) and reused by `wm_maze_train.py`.
- Added focused regression tests in `wm_refactor_tests.py`:
  - predictor cache/reset behavior
  - rollout contract validation
  - MazeVecEnv + OnPolicyTrainer + JointWM startup smoke run

Next:
1. Keep baseline-only joint runs as primary path (`rnn`/`transformer`) with intrinsic reward only.
2. After baseline behavior is stable, extend world-model joint path to latent models (`rssm`/`tssm`) under the same rollout contract.

## Thu Apr 16 AM UTC 2026

I can make baselines more like world models with discrete states by projecting ht -> zt

## Update (2026-04-19, Shared Extraction + Contrastive/Policy Gradient Cleanup)

Completed:
- Shared observation extraction/validation now lives in base mixin:
  - added shared `_validate_obs_contract` + `_shift_prev_actions`
  - `rssm`/`tssm` now consume the same path via `_encode_obs(...)` as baselines
- Rollout contract now supports `actions=None` when `prev_actions` is provided; policy rollout caller updated to pass `actions=None` in rollout mode.
- Contrastive target detach is now centralized:
  - baseline target projector detaches internally
  - world-model (`z`) target projector detaches internally
- Simplified contrastive aux wiring:
  - removed per-forward `if contrastive_head is not None` guards
  - always populate contrastive aux keys; projector helpers return `None`/`[]` when disabled.
- Moved `detach_action_heads` from forward-time arg to model construction setting (default `True`):
  - removed per-call `detach_action_heads=...` overrides
  - forward paths now read `self.detach_action_heads`.
- Joint WM+policy path now explicitly enables action-head gradient flow by setting `model.detach_action_heads = False` in `create_maze_world_model(...)`.

## Update (2026-04-19, Learning-Progress Reward Plan - Lightweight Variant)

Goal:
- Use learning-progress (LP) intrinsic reward without expensive twin-model cross-fit.

Rationale:
- Full cross-fit (`m1/m2`, `D1/D2`) is cleaner but too expensive for fast online iteration.
- Cheaper approximation: measure CPC error reduction on current policy data, and reduce overfit by mixing replay in training.

Plan:
1. Build datasets per update:
   - `D_new`: newly accumulated policy episodes.
   - `D_mix`: `D_new` + sampled replay episodes.
2. Pre-eval on `D_new` (no grad):
   - compute per-step AC-CPC error `e_before` (e.g. `-log p_pos`).
3. Train world model on `D_mix` (normal WM objective, including CPC).
4. Post-eval on `D_new` (no grad):
   - compute per-step AC-CPC error `e_after`.
5. LP reward on `D_new` steps:
   - `r_lp = e_before - e_after`.
6. Policy reward:
   - `r_total = env_reward_scale * r_env + lp_reward_scale * r_lp`.
7. Stabilization:
   - clip LP reward, apply running normalization, and keep meaningful replay fraction in `D_mix`.

Notes:
- LP reward is computed only for `D_new` policy steps (not replay steps).
- This is biased versus held-out cross-fit, but much cheaper and appropriate for first experiments.

## Update (2026-04-19, Joint Trainer Simplification + Run Script)

- Added new launcher script with intrinsic-only defaults for maze joint training:
  - `wm_train_maze_joint_default.sh`
  - default uses baseline RNN + AC-CPC, `env_reward_scale=0.0`, replay-enabled WM updates.

## update Thu Apr 23 09:27:15 AM UTC 2026

now using cpc error before training as surprise reward
total reward is error before + lp = error before + (error before - error after)

now using sliding window instead of full batches for cpc loss
added computation of maze coverage to rl maze env
