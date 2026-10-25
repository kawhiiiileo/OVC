# Outcome-Conditioned Visual Credit

OVC separates relative visual association from outcome-conditioned token credit.
Its components are Relative Visual Encoding (RVE), Outcome-conditioned Evidence
Calibration (OEC), and Normalized Credit Allocation (NCA).

## Relative Visual Encoding

At a selected intermediate layer, RVE contrasts each predictive state's cosine
affinity to image and non-image prompt prototypes under a shared diagonal scale.
It combines the resulting coordinate with the prediction-aware feature
`v = (1 - p_token) * h`. The visual coordinate itself is not multiplied by
`1 - p_token`.

## Outcome-conditioned Evidence Calibration

OEC estimates advantage-magnitude-weighted positive and negative references,
uses a regularized diagonal scale of within-side variation, and forms a signed
distance contrast. One soft reference refinement gives the final references.
Advantage signs indicate relative response reward, not token correctness.

## Normalized Credit Allocation

NCA converts sign-conditioned evidence into positive token weights and normalizes
their mean over the selected statistical batch. The weighted advantage is
`A_prime = stop_gradient(weight) * A`. Weights preserve advantage signs while
modulating individual token-gradient contributions.

## Policy integration

GSPO-token retains a sequence-level forward ratio and clipping, with token-local
gradients:

```python
log_ratio = log_prob - log_prob.detach() + sequence_log_ratio.detach().unsqueeze(-1)
```

Credit quantities are detached; policy log probabilities remain differentiable.
Behavior-policy log probabilities stay fixed during the update.

The released recipe uses Qwen3.5-9B, GSPO-token and hidden layer 16. RVE is in
`verl/workers/utils/delta_visual_grounding.py`; calibration and allocation are in
`verl/workers/utils/losses.py`. Shared tensor operations live in
`ovc_release/credit/`. See [reproduction](reproduction.md) for training setup.
