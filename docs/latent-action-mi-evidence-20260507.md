# Latent Action MI Evidence - 2026-05-07

This note records the diagnostic evidence for whether the factorized
hyperbolic/RadProg LAM places action-relevant information in the quantized
latent action, especially in the radius/norm component.

## Question

The working claim is not just that the factorized LAM uses different token IDs.
The claim is:

```text
RadProg/hyperbolic factorization makes the quantized latent action geometry more
action-informative, and part of that information is carried by radial magnitude.
```

Therefore, the analysis should use `z_q` itself, not only token IDs. It should
also avoid destroying radius/norm information unless the goal is a conservative
control.

## Artifacts

Feature cache:

```text
outputs/analysis/cache/zq_full_action_k9_all_2048/
```

Models in the cache:

```text
factorized_hyper_30k.npz
euclidean_visual_vq_30k.npz
univla_stage2_60k.npz
```

Main KSG output:

```text
outputs/analysis/zq_full_action_ksg_xproj_yzscore_k9_all_2048.csv
```

Radius ablation output:

```text
outputs/analysis/hyper_no_radius_ksg_k9_all_2048.csv
```

The cache is a fast diagnostic cache using forced `k=9` horizon and
`ignore-valid` sampling. Treat these numbers as evidence for direction, not as
final paper numbers. For paper use, repeat with a strict-valid cache and
multiple random seeds.

## Metric Choice

Recommended main metric:

```text
X = raw_global(z_q sequence)
Y = per-dimension zscore(action sequence)
KSG k = 5
samples = 1024
X projection dims = 16, 32, 64, 128, 256
```

Why this metric is the best fit for the current claim:

- Ground-truth action dimensions have different units and variances, so action
  should be per-dimension z-scored.
- The factorized hyperbolic model is designed to put information in the
  radius/norm of `z_q`, so per-dimension z-scoring of `z_q` can erase part of
  the signal being tested.
- `raw_global(z_q)` removes arbitrary global model scale with one scalar
  mean/std while preserving sample-level radial structure.

Conservative control:

```text
X = per-dimension zscore(z_q sequence)
Y = per-dimension zscore(action sequence)
```

This tests whether the factorized hyperbolic model still contains action
information after weakening radial scale. It should be reported as a control,
not the main RadProg-aligned metric.

Avoid using as the main result:

```text
X = raw(z_q sequence)
Y = raw(action sequence)
```

This can overstate differences because finite-sample KSG is sensitive to raw
scale and coordinate variance even though true mutual information is invariant
to invertible rescalings.

## Main Results

All rows below use:

```text
Y = zscore(action sequence, 70D)
KSG k = 5
samples = 1024
```

```text
dim | hyper full raw_global | hyper full zscore | hyper no-radius token_unit | hyper no-radius sample_unit | euclid raw_global | euclid zscore | univla raw_global | univla zscore
16  | 0.426 | 0.251 | 0.388 | 0.311 | 0.173 | 0.143 | 0.239 | 0.190
32  | 0.529 | 0.283 | 0.368 | 0.289 | 0.224 | 0.190 | 0.331 | 0.294
64  | 0.644 | 0.447 | 0.330 | 0.279 | 0.365 | 0.274 | 0.392 | 0.340
128 | 0.766 | 0.618 | 0.306 | 0.263 | 0.591 | 0.510 | 0.434 | 0.417
256 | 0.836 | 0.706 | 0.290 | 0.259 | 0.732 | 0.659 | 0.475 | 0.424
```

At the recommended main metric, `raw_global(z_q) -> zscore(action)`, the
factorized hyperbolic model beats both baselines at every projection dimension:

```text
dim 16:  hyper 0.426 | euclid 0.173 | univla 0.239
dim 32:  hyper 0.529 | euclid 0.224 | univla 0.331
dim 64:  hyper 0.644 | euclid 0.365 | univla 0.392
dim 128: hyper 0.766 | euclid 0.591 | univla 0.434
dim 256: hyper 0.836 | euclid 0.732 | univla 0.475
```

At the conservative control, `zscore(z_q) -> zscore(action)`, the factorized
hyperbolic model still beats both baselines, but the margin over the Euclidean
visual VQ baseline is smaller:

```text
dim 16:  hyper 0.251 | euclid 0.143 | univla 0.190
dim 32:  hyper 0.283 | euclid 0.190 | univla 0.294
dim 64:  hyper 0.447 | euclid 0.274 | univla 0.340
dim 128: hyper 0.618 | euclid 0.510 | univla 0.417
dim 256: hyper 0.706 | euclid 0.659 | univla 0.424
```

The 32D conservative control is the only dim in this table where UniVLA stage2
is slightly above factorized hyperbolic. The 256D conservative setting still
favours factorized hyperbolic.

## Radius Ablation

The radius ablation removes radial magnitude from the factorized hyperbolic
`z_q` and compares it to the full factorized latent action.

```text
dim | full hyper raw_global | no-radius token_unit | no-radius sample_unit
16  | 0.444 | 0.388 | 0.311
32  | 0.563 | 0.368 | 0.289
64  | 0.685 | 0.330 | 0.279
128 | 0.785 | 0.306 | 0.263
256 | 0.825 | 0.290 | 0.259
```

The most important diagnostic is the 256D comparison:

```text
full hyper raw_global: 0.825
no-radius token_unit: 0.290
no-radius sample_unit: 0.259
```

Interpretation:

```text
Removing radial magnitude substantially reduces z_q -> action MI.
```

This supports the claim that the radial part of the factorized hyperbolic
latent action carries action-relevant information. It is not just a decorative
scale multiplier.

## Wording To Use

Use:

```text
We globally standardize latent actions to remove arbitrary model-scale
differences while preserving radial information, and per-dimension standardize
actions to remove action-unit imbalance.
```

Use:

```text
The full factorized hyperbolic latent action has higher KSG MI with the
ground-truth action sequence than Euclidean visual VQ and UniVLA stage2 under a
geometry-preserving normalization.
```

Use:

```text
Removing radial magnitude from the factorized hyperbolic latent action sharply
reduces MI, indicating that RadProg places action-relevant information in the
radius/norm component.
```

Avoid:

```text
The raw/raw KSG score proves hyperbolic is better.
```

That statement is too easy to attack because finite-sample KSG is sensitive to
raw coordinate scale.

## Next Checks Before Paper Use

- Repeat the main metric with seeds `0..4` and report mean plus std.
- Rebuild a strict-valid `k=9` cache instead of using `ignore-valid`.
- Report both `raw_global -> zscore(action)` and `zscore -> zscore(action)`.
- Keep the radius ablation table next to the main KSG table.
- Optionally add direct scalar diagnostics:
  `MI(||z_q||; ||action_seq||)` and
  `MI(||z_q||; state-change norm)`.
