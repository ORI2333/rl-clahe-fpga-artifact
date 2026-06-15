# A25 Algorithm-Choice Diagnostic

## Scope
- Pure-Python diagnostic based on saved SAC teacher trajectory labels.
- Supports the SAC rationale by quantifying the continuous bounded action formulation.
- Not a head-to-head retraining benchmark for PPO, DQN, and TD3.

## Action-Space Statistics
- Transitions: 3000
- Teacher action range: -1.9965 to 1.9972
- Mean absolute action: 0.9272
- p95 absolute action: 1.8615
- Rounded-to-0.001 unique action values: 1762

## DQN-Style Discretization Error
| Action bins | Step | Mean | p95 | Max | Mean final-CL error | p95 final-CL error | Max final-CL error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.0000 | 0.2494 | 0.4742 | 0.5000 | 0.3997 | 1.1790 | 1.8039 |
| 9 | 0.5000 | 0.1252 | 0.2374 | 0.2498 | 0.2087 | 0.6039 | 0.9905 |
| 17 | 0.2500 | 0.0634 | 0.1193 | 0.1250 | 0.1072 | 0.3081 | 0.4830 |
| 33 | 0.1250 | 0.0309 | 0.0594 | 0.0625 | 0.0505 | 0.1467 | 0.2214 |
| 65 | 0.0625 | 0.0156 | 0.0297 | 0.0312 | 0.0261 | 0.0717 | 0.1114 |

## Interpretation For Response Drafting
- DQN is not impossible, but a discrete action grid introduces quantization error in every clip-limit update unless many bins are used.
- PPO and TD3 remain plausible continuous-control alternatives; this diagnostic only supports why a continuous off-policy method is a natural fit.
- SAC is consistent with the bounded continuous action and the repeated offline image-enhancement environment where replay-based sample reuse is useful.
- Use this as supporting formulation evidence, not as a claim that SAC universally outperforms PPO, DQN, or TD3.

## SAC Training Logs Present Locally
| Log | Points | First step | Last step | First value | Last value | Last-20 mean | Last-20 std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC_lite_re_rew_mean.csv | 1000 | 400 | 359025 | -1.0993 | 2.6777 | 2.5681 | 0.5621 |
| SAC_rich_re_rew_mean.csv | 1000 | 870 | 600000 | -8.9247 | 7.1848 | 7.8976 | 1.2762 |

## Evidence Files
- `action_space_summary.csv`
- `dqn_discretization_summary.csv`
- `sac_training_log_summary.csv`

The diagnostic uses saved trajectory-label summaries. It does not redistribute
the full training dataset or teacher checkpoints.
