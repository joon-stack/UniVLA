# UniVLA Hyperbolic LAM

This context covers the visual VQ latent action model experiments that shape Bridge latent actions with RadProg and hyperbolic geometry.

## Language

**Latent Action Model**:
A model that encodes visual transitions into compact latent actions used to reconstruct future visual features.
_Avoid_: Policy, VLA head

**VQ Embedding**:
The continuous pre-quantization latent emitted by the LAM encoder after any configured prelift transform.
_Avoid_: Raw action, codebook output

**Codebook Vector**:
The selected VQ prototype for a VQ embedding.
_Avoid_: Encoder output

**Quantized Latent Action**:
The straight-through VQ output whose forward value is the selected codebook vector and whose backward path reaches the encoder.
_Avoid_: Raw z, unquantized action

**Factorized Quantized Latent Action**:
A quantized latent action represented by one radius token plus direction tokens. In the current Bridge VLA handoff, each transition uses one radius token and four direction tokens.
_Avoid_: Single flat VQ token

**RadProg Target**:
The latent representation used by the RadProg radial and progress losses.
_Avoid_: Generic latent

**FDM Input**:
The latent action sequence passed into the forward dynamics decoder path.
_Avoid_: RadProg-only latent

**VLA LAM Handoff**:
The boundary where VLA pretraining loads a trained LAM checkpoint and its matching config snapshot to produce latent action tokens for Bridge trajectories.
_Avoid_: LAM training run

**Dead Code Restart**:
Periodic reinitialization of VQ codes that have zero recorded usage during a restart window.
_Avoid_: Codebook reset

**Geometry-Preserving MI Analysis**:
An analysis protocol for measuring action information in latent actions while
preserving sample-level radial or norm structure. In the current Bridge
diagnostics, this means globally standardizing the latent action vector with one
scalar mean/std and per-dimension standardizing the ground-truth action vector.
_Avoid_: Raw MI, scale-free MI

**Radius Ablation**:
An ablation that removes the factorized latent action's radial magnitude and
measures whether direction-only latent actions retain the same action
information.
_Avoid_: No-hyperbolic baseline

## Relationships

- A **VQ Embedding** is assigned to exactly one **Codebook Vector** per latent action token.
- A **Quantized Latent Action** is derived from a **VQ Embedding** and a **Codebook Vector** through straight-through estimation.
- The hyperbolic visual VQ experiment treats the **VQ Embedding** as the **RadProg Target** and the **Quantized Latent Action** as the **FDM Input**.
- The factorized **VLA LAM Handoff** consumes a checkpoint plus a config snapshot. The current token contract is one radius token plus four direction tokens, `latent_action_token_len=5`, and `codebook_size=32`.
- **Dead Code Restart** is delayed to a 30000-step window for the current hyperbolic visual VQ experiment.
- A **Geometry-Preserving MI Analysis** is appropriate when the claim is that
  RadProg places action-relevant information in the norm/radius of the
  **Quantized Latent Action**.
- A **Radius Ablation** compares the full factorized **Quantized Latent Action**
  against a direction-only version of the same action.

## Example dialogue

> **Dev:** "Should RadProg read the encoder embedding or the action consumed by the decoder?"
> **Domain expert:** "Use the **VQ Embedding** for the regularizer, then verify that the **Quantized Latent Action** preserves the geometry when quantization error is low."

## Flagged ambiguities

- "z_q" means **Quantized Latent Action**, not a norm-preserving transform of the **VQ Embedding**.
- "emb" means **VQ Embedding**, not the selected **Codebook Vector**.
- In the factorized handoff, `codebook_size=32` means the VLA latent-action token vocabulary covers both radius and direction token IDs. It does not mean the LAM is using 32 flat VQ prototypes.
- "reset" in these experiments means **Dead Code Restart**, not restarting the full training job.
- Per-dimension z-scoring of latent actions is a conservative control, but it
  weakens radial information. It should not be the only metric for a RadProg
  claim.
