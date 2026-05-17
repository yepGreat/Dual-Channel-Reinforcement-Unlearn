# DFRU — Dual-Channel Fine-Grained Reinforcement Unlearning

*Integrating Reward Shaping and State Perturbation for Selective Forgetting.*

Code release for **DFRU**, a unified unlearning method for selective object forgetting in deep
reinforcement learning on grid-world navigation. DFRU fuses the two single-axis methods
**SPU** (adversarial reward) and **SSP** (targeted state poisoning) into a single training
procedure: during the unlearning phase the observation around the forgotten obstacle is poisoned
*and* a positive adversarial reward is applied on contact.

DFRU replaces SPU and SSP as the unlearning algorithm; the rest of the pipeline (Pretrain /
Retrain baselines, map generation, verification, plotting) is unchanged.

---

## 1. Methods

| Name      | Role                                                                                  | Script               |
| --------- | ------------------------------------------------------------------------------------- | -------------------- |
| Pretrain  | DQN policy trained on the full map (starting point for unlearning).                   | `SPU_script.py` *    |
| Retrain   | DQN policy trained from scratch on the *clean* map (target-type removed). Oracle baseline. | `base_line_model.py` |
| **DFRU**  | Joint reward-side (`reward_scale`) and observation-side (`poison_intensity`) unlearning. | `dfru.py`            |

\* `SPU_script.py` is retained only for the pretraining stage; the SPU/SSP unlearning paths are
superseded by DFRU.

Default experimental setup:

- Environment: `grid_world`, size `10 × 10`, 4 discrete actions, 10-d state (agent xy + 8 neighbors).
- Obstacle types: `type_1` … `type_4`, plus boundary obstacles.
- Number of maps: `50`, default seed `28`.
- Forgotten type: `type_3`. Forget-set ratio: `30%` of maps.

---

## 2. Repository layout

```text
DFRU/
├─ README.md
├─ utils.py                     # Env, DQN, replay buffer, model I/O, reward functions
├─ init.py                      # Map / clean-map / unlearn-flag generation, module init
├─ Hyperparameter_control.py    # Shared hyperparameters and seed control
├─ SPU_script.py                # Pretrain (train / test) — entry kept for the Pretrain stage
├─ base_line_model.py           # Retrain baseline (train / test)
├─ dfru.py                      # DFRU unlearning (train / test)
│
├─ batch_train_pretrain_models.py
├─ batch_test_pretrain_models.py
├─ get_best_pretrain_model.py
│
├─ batch_train_retrain_models.py
├─ batch_test_retrain_models.py
│
├─ batch_dfru_training_with_diff_steps.py
├─ batch_verify_dfru.py
│
├─ plot_dfru_unlearn_retain.py        # Per-method curves (unlearn vs retain set)
├─ plot_dfru_heatmap.py               # 2D heatmap over (reward_scale, poison_intensity)
└─ plot_dfru_3d_surface.py            # 3D surface over (reward_scale, poison_intensity)
```

Generated artefacts (created on first run):

- `map_data/grid_world/` — maps, clean maps, and unlearn flags (`.json` / `.pkl`).
- `model/grid_world/DFRU_Model_standard_epoch_*/tem_dfru_max_steps_*/` — DFRU checkpoints.
- `result/grid_world/` — verification CSV/XLSX and plotted figures.

---

## 3. Installation

Python `3.10+` is recommended. Install PyTorch matching your CUDA version, then:

```bash
pip install torch gym numpy pandas matplotlib tqdm openpyxl pillow
```

---

## 4. Reproducing the experiments

The pipeline is split into stages; run them in order. All scripts accept `-h` for full options.

### 4.1 Pretrain baseline

```bash
python batch_train_pretrain_models.py
python batch_test_pretrain_models.py
python get_best_pretrain_model.py
```

This sweeps seeds / training steps and tags the best checkpoint per seed as `_best`.
The main-line experiments use `seed=28`.

### 4.2 Retrain baseline (clean-map oracle)

```bash
python batch_train_retrain_models.py
python batch_test_retrain_models.py
```

### 4.3 DFRU — unlearning

```bash
python batch_dfru_training_with_diff_steps.py
python batch_verify_dfru.py
```

Key hyperparameters (see `dfru.py`):

- `--reward_scale` — magnitude of the positive adversarial reward on the forgotten type (SPU axis).
- `--poison_intensity` — strength of the targeted observation perturbation (SSP axis).
- `--tem_dfru_max_steps` — max steps per episode during the unlearning phase.
- `--unlearn_epoch`, `--unlearn_maps_ratio`, `--unlearn_obstacles_type`.

### 4.4 Plots

```bash
python plot_dfru_unlearn_retain.py     # unlearn vs retain set
python plot_dfru_heatmap.py            # 2D heatmap over (reward_scale, poison_intensity)
python plot_dfru_3d_surface.py         # 3D surface over (reward_scale, poison_intensity)
```

---

## 5. Evaluation metrics

`batch_verify_dfru.py` reports, separately on the `all` / `unlearn` / `retain` splits:

- `avg_steps` — average episode length.
- `avg_reward` — average cumulative reward.
- `collision_summary` — per-type collision counts.
- `unlearn_ratio` — fraction of collisions on the forgotten type (selective-forgetting score).
- `avg_perplexity` — decision uncertainty in the neighborhood of the target obstacle.

This makes the trade-off between *forgetting* (high collision rate on the unlearn set) and
*retention* (unchanged behavior on the retain set) directly comparable across hyperparameter
settings.

---

## 6. Notes

This is research code released alongside the paper. The DFRU plotting scripts (heatmap / 3D surface)
hard-code the swept ranges of `reward_scale` and `poison_intensity` used to produce the paper
figures; adjust them as needed for your own experiments. Maps and unlearn-flag files are produced
deterministically from `--seed`, so results should be reproducible once the pretraining seed is
fixed.
