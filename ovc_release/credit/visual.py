# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class PromptGroupSigmaAccumulator:
    """Accumulate exact prompt-group whitening moments across micro-batches."""

    def __init__(self, *, rollout_n: int, sigma_eps: float, sequence_chunk_size: int = 512):
        self.rollout_n = max(int(rollout_n), 1)
        self.sigma_eps = float(sigma_eps)
        self.sequence_chunk_size = max(int(sequence_chunk_size), 1)
        self._sample_counts: dict[str, int] = {}
        self._moments: dict[tuple[str, str], dict[str, object]] = {}

    @staticmethod
    def _uid(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _merge(state: dict[str, object] | None, values: torch.Tensor) -> dict[str, object]:
        count = int(values.size(0))
        mean = values.mean(dim=0)
        centered = values - mean
        m2 = centered.square().sum(dim=0)
        if state is None:
            return {"count": count, "mean": mean, "m2": m2}

        old_count = int(state["count"])
        total = old_count + count
        old_mean = state["mean"]
        delta = mean - old_mean
        merged_mean = old_mean + delta * (float(count) / float(total))
        merged_m2 = state["m2"] + m2 + delta.square() * (float(old_count * count) / float(total))
        return {"count": total, "mean": merged_mean, "m2": merged_m2}

    def update(
        self,
        response: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor | None,
        prompt_uids,
    ) -> None:
        if response.dim() != 3 or response_mask.shape != response.shape[:2]:
            raise ValueError("response/mask must be [B,S,H]/[B,S]")
        if len(prompt_uids) != response.size(0):
            raise ValueError(f"prompt UID count mismatch: {len(prompt_uids)} != {response.size(0)}")

        valid = response_mask.to(device=response.device, dtype=torch.bool)
        adv = advantages.to(device=response.device, dtype=torch.float32) if advantages is not None else None
        for row, raw_uid in enumerate(prompt_uids):
            uid = self._uid(raw_uid)
            self._sample_counts[uid] = self._sample_counts.get(uid, 0) + 1
            class_name = None
            if adv is not None:
                if adv[row] > 0:
                    class_name = "pos"
                elif adv[row] < 0:
                    class_name = "neg"
            for start in range(0, response.size(1), self.sequence_chunk_size):
                stop = min(start + self.sequence_chunk_size, response.size(1))
                chunk_mask = valid[row, start:stop]
                if not chunk_mask.any():
                    continue
                values = response[row, start:stop].detach().to(torch.float32)[chunk_mask]
                all_key = (uid, "all")
                self._moments[all_key] = self._merge(self._moments.get(all_key), values)
                if class_name is not None:
                    class_key = (uid, class_name)
                    self._moments[class_key] = self._merge(self._moments.get(class_key), values)

    def finalize(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if not self._sample_counts:
            raise RuntimeError("prompt-group whitening received no samples")
        bad = {uid: count for uid, count in self._sample_counts.items() if count != self.rollout_n}
        if bad:
            preview = ", ".join(f"{uid}:{count}" for uid, count in list(bad.items())[:8])
            raise RuntimeError(
                f"prompt-group whitening requires exactly rollout_n={self.rollout_n} samples per UID; got {preview}"
            )

        sigmas = {}
        for uid in self._sample_counts:
            class_states = [self._moments.get((uid, name)) for name in ("pos", "neg")]
            class_states = [state for state in class_states if state is not None]
            if class_states:
                count = sum(int(state["count"]) for state in class_states)
                m2 = torch.stack([state["m2"] for state in class_states]).sum(dim=0)
            else:
                state = self._moments.get((uid, "all"))
                if state is None:
                    raise RuntimeError(f"prompt UID {uid} has no valid response tokens")
                count = int(state["count"])
                m2 = state["m2"]
            pooled_var = m2 / max(float(count), 1.0)
            sigmas[uid] = torch.sqrt(
                pooled_var.clamp_min(self.sigma_eps * self.sigma_eps)
            ).detach()

        counts = list(self._sample_counts.values())
        audit = {
            "prompt_groups": float(len(counts)),
            "group_size_min": float(min(counts)),
            "group_size_max": float(max(counts)),
            "group_complete": 1.0,
        }
        return sigmas, audit


def sigma_for_prompt_uids(
    sigma_by_uid: dict[str, torch.Tensor], prompt_uids, *, device: torch.device
) -> torch.Tensor:
    rows = []
    for raw_uid in prompt_uids:
        uid = PromptGroupSigmaAccumulator._uid(raw_uid)
        if uid not in sigma_by_uid:
            raise RuntimeError(f"missing prompt-group sigma for UID {uid}")
        rows.append(sigma_by_uid[uid].to(device=device))
    if not rows:
        raise RuntimeError("cannot build prompt-group sigma for an empty micro-batch")
    return torch.stack(rows, dim=0)



def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denom


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return (a * b).sum(dim=-1) / denom.clamp_min(float(eps))


def _response_sigma(
    response: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor | None,
    *,
    rollout_n: int,
    sigma_eps: float,
) -> torch.Tensor:
    """Current-batch diagonal std for whitening, computed under no-grad."""

    B, _, H = response.shape
    valid = response_mask.to(device=response.device, dtype=torch.bool)
    sigma = response.new_full((B, H), float(sigma_eps), dtype=torch.float32)
    group = max(int(rollout_n), 1)
    adv = advantages.to(device=response.device, dtype=torch.float32) if advantages is not None else None
    resp = response.detach().to(torch.float32)

    for start in range(0, B, group):
        end = min(start + group, B)
        g_valid = valid[start:end]
        if not g_valid.any():
            continue
        g_resp = resp[start:end]
        pooled_var = None
        if adv is not None:
            g_adv = adv[start:end]
            pos = g_valid & g_adv.gt(0).unsqueeze(1)
            neg = g_valid & g_adv.lt(0).unsqueeze(1)
            parts = []
            counts = []
            for cls_mask in (pos, neg):
                count = cls_mask.sum()
                if count <= 0:
                    continue
                vals = g_resp.masked_select(cls_mask.unsqueeze(-1)).view(-1, H)
                mean = vals.mean(dim=0)
                parts.append(((vals - mean) ** 2).sum(dim=0))
                counts.append(count.to(torch.float32))
            if parts:
                pooled_var = torch.stack(parts).sum(dim=0) / torch.stack(counts).sum().clamp_min(1.0)
        if pooled_var is None:
            vals = g_resp.masked_select(g_valid.unsqueeze(-1)).view(-1, H)
            pooled_var = vals.var(dim=0, unbiased=False) if vals.size(0) > 1 else torch.zeros(H, device=response.device)
        sigma[start:end] = torch.sqrt(pooled_var.clamp_min(float(sigma_eps) * float(sigma_eps)))

    return sigma


def compute_visual_grounding_rho(
    full_hidden: torch.Tensor,
    response_hidden: torch.Tensor,
    image_token_mask: torch.Tensor,
    response_token_mask: torch.Tensor | None = None,
    *,
    text_token_mask: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    rollout_n: int = 1,
    signal_version: str = "v1",
    whiten: str = "diag",
    text_baseline: bool = True,
    subspace_rank: int = 0,
    sigma_eps: float = 1.0e-6,
    normalize_rho: bool = False,
    cos_eps: float = 1.0e-6,
    response_sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute visual grounding signal from already captured hidden states.

    signal_version="v1" preserves the original rho = cos(h, mu_v).  "v2"
    upgrades rho to rho* by optional diagonal whitening and subtracting a
    prompt-text baseline. All outputs are detached from the graph.
    """

    with torch.no_grad():
        if full_hidden.dim() != 3 or response_hidden.dim() != 3:
            raise ValueError("full_hidden and response_hidden must be [B, S, H] tensors")
        if image_token_mask.shape != full_hidden.shape[:2]:
            raise ValueError("image_token_mask must match full_hidden batch and sequence dimensions")

        version = str(signal_version or "v1").lower()
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported visual grounding signal_version: {signal_version}")
        whiten_mode = str(whiten or "diag").lower()
        if whiten_mode not in {"diag", "none"}:
            raise ValueError(f"Unsupported visual grounding v2.whiten: {whiten}")

        full = full_hidden.detach().to(torch.float32)
        response = response_hidden.detach().to(torch.float32)
        device = response.device
        image_mask = image_token_mask.to(device=device, dtype=torch.bool)

        if response_token_mask is None:
            response_mask = torch.ones(response.shape[:2], device=device, dtype=torch.bool)
        else:
            response_mask = response_token_mask.to(device=device, dtype=torch.bool)

        image_counts = image_mask.sum(dim=1)
        has_image = image_counts > 0
        mu_v = _masked_mean(full, image_mask, dim=1)

        response_for_score = response
        mu_v_for_score = mu_v
        mu_txt_for_score = None
        if version == "v2":
            if whiten_mode == "diag":
                if response_sigma is None:
                    sigma = _response_sigma(
                        response,
                        response_mask,
                        advantages,
                        rollout_n=rollout_n,
                        sigma_eps=sigma_eps,
                    )
                else:
                    if response_sigma.shape != (response.size(0), response.size(2)):
                        raise ValueError(
                            "response_sigma must match response hidden batch/feature dimensions"
                        )
                    sigma = response_sigma.detach().to(device=device, dtype=torch.float32)
                response_for_score = response / sigma.unsqueeze(1).clamp_min(float(sigma_eps))
                mu_v_for_score = mu_v / sigma.clamp_min(float(sigma_eps))
            if text_baseline and text_token_mask is not None:
                txt_mask = text_token_mask.to(device=device, dtype=torch.bool)
                txt_counts = txt_mask.sum(dim=1)
                if txt_counts.gt(0).any():
                    mu_txt = _masked_mean(full, txt_mask, dim=1)
                    if whiten_mode == "diag":
                        mu_txt = mu_txt / sigma.clamp_min(float(sigma_eps))
                    mu_txt_for_score = mu_txt

        if version == "v2" and int(subspace_rank or 0) > 0:
            rank = int(subspace_rank)
            rho = response.new_zeros(response.shape[:2], dtype=torch.float32)
            for i in range(response.size(0)):
                if not has_image[i]:
                    continue
                img = full[i, image_mask[i]]
                resp_i = response_for_score[i]
                if whiten_mode == "diag":
                    img = img / sigma[i].clamp_min(float(sigma_eps))
                if img.size(0) < 2:
                    vis_score = _cosine(resp_i, mu_v_for_score[i].unsqueeze(0).expand_as(resp_i), cos_eps)
                else:
                    q = min(rank, img.size(0), img.size(1))
                    try:
                        U, _, _ = torch.svd_lowrank(img, q=q)
                        proj = torch.matmul(resp_i, U[:, :q])
                        vis_score = torch.linalg.vector_norm(proj, dim=-1) / torch.linalg.vector_norm(resp_i, dim=-1).clamp_min(float(cos_eps))
                    except RuntimeError:
                        vis_score = _cosine(resp_i, mu_v_for_score[i].unsqueeze(0).expand_as(resp_i), cos_eps)
                if mu_txt_for_score is not None:
                    vis_score = vis_score - _cosine(resp_i, mu_txt_for_score[i].unsqueeze(0).expand_as(resp_i), cos_eps)
                rho[i] = vis_score
        else:
            rho = _cosine(response_for_score, mu_v_for_score.unsqueeze(1).expand_as(response_for_score), cos_eps)
            valid_cos = has_image.unsqueeze(1) & torch.linalg.vector_norm(response_for_score, dim=-1).gt(0) & torch.linalg.vector_norm(mu_v_for_score, dim=-1).gt(0).unsqueeze(1)
            rho = torch.where(valid_cos, rho, torch.zeros_like(rho))
            if version == "v2" and mu_txt_for_score is not None:
                txt_cos = _cosine(response_for_score, mu_txt_for_score.unsqueeze(1).expand_as(response_for_score), cos_eps)
                valid_txt = torch.linalg.vector_norm(mu_txt_for_score, dim=-1).gt(0).unsqueeze(1)
                rho = torch.where(valid_txt, rho - txt_cos, rho)

        if normalize_rho:
            rho = 0.5 * (rho + 1.0)

        image_response_mask = has_image.unsqueeze(1) & response_mask
        if image_response_mask.any():
            neutral = rho.masked_select(image_response_mask).mean()
        else:
            neutral = rho.new_zeros(())
        rho = torch.where(has_image.unsqueeze(1), rho, neutral.expand_as(rho))
        rho = torch.where(response_mask, rho, torch.zeros_like(rho))
        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

        return rho.detach(), image_response_mask.detach(), has_image.detach()


def append_rho_coordinate(v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return torch.cat([v, rho.to(device=v.device, dtype=v.dtype).unsqueeze(-1)], dim=-1)


def weighted_sum_with_optional_rho_mask(
    weights: torch.Tensor,
    values: torch.Tensor,
    image_response_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Sum weights * values, masking only the final rho coordinate if present."""

    if image_response_mask is None:
        return (weights.unsqueeze(-1) * values).sum(dim=(0, 1))

    base = (weights.unsqueeze(-1) * values[..., :-1]).sum(dim=(0, 1))
    rho_weights = weights * image_response_mask.to(device=weights.device, dtype=weights.dtype)
    rho = (rho_weights * values[..., -1]).sum().view(1)
    return torch.cat([base, rho])


def weighted_sumsq_with_optional_rho_mask(
    weights: torch.Tensor,
    values: torch.Tensor,
    image_response_mask: torch.Tensor | None,
) -> torch.Tensor:
    return weighted_sum_with_optional_rho_mask(weights, values * values, image_response_mask)


def denominator_with_optional_rho_mask(
    weights: torch.Tensor,
    values: torch.Tensor,
    image_response_mask: torch.Tensor | None,
) -> torch.Tensor:
    if image_response_mask is None:
        return weights.sum()

    base_den = weights.sum().expand(values.size(-1) - 1)
    rho_den = (weights * image_response_mask.to(device=weights.device, dtype=weights.dtype)).sum().view(1)
    return torch.cat([base_den, rho_den])


def update_rho_metric_sums(
    metric_sums: dict[str, torch.Tensor] | None,
    rho: torch.Tensor,
    image_response_mask: torch.Tensor,
    advantages: torch.Tensor,
    response_token_mask: torch.Tensor,
    *,
    rollout_n: int = 1,
    kappa: float = 1.0,
    clip_c: float = 0.2,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        rho_f = rho.detach().to(torch.float32)
        valid = image_response_mask.to(device=rho.device, dtype=torch.bool) & response_token_mask.to(
            device=rho.device, dtype=torch.bool
        )
        adv = advantages.to(device=rho.device, dtype=torch.float32)
        pos = valid & adv.gt(0).unsqueeze(1)
        neg = valid & adv.lt(0).unsqueeze(1)

        if metric_sums is None:
            metric_sums = {
                "sum": rho_f.new_zeros(()),
                "sumsq": rho_f.new_zeros(()),
                "count": rho_f.new_zeros(()),
                "pos_sum": rho_f.new_zeros(()),
                "pos_count": rho_f.new_zeros(()),
                "neg_sum": rho_f.new_zeros(()),
                "neg_count": rho_f.new_zeros(()),
                "early_sum": rho_f.new_zeros(()),
                "early_count": rho_f.new_zeros(()),
                "late_sum": rho_f.new_zeros(()),
                "late_count": rho_f.new_zeros(()),
                "pg_diff_sum": rho_f.new_zeros(()),
                "pg_fisher_sum": rho_f.new_zeros(()),
                "pg_count": rho_f.new_zeros(()),
                "gamma_sum": rho_f.new_zeros(()),
                "gamma_abs_sum": rho_f.new_zeros(()),
                "gamma_count": rho_f.new_zeros(()),
                "clip_sum": rho_f.new_zeros(()),
                "clip_count": rho_f.new_zeros(()),
            }

        valid_f = valid.to(rho_f.dtype)
        metric_sums["sum"] += (rho_f * valid_f).sum()
        metric_sums["sumsq"] += (rho_f * rho_f * valid_f).sum()
        metric_sums["count"] += valid_f.sum()

        pos_f = pos.to(rho_f.dtype)
        neg_f = neg.to(rho_f.dtype)
        metric_sums["pos_sum"] += (rho_f * pos_f).sum()
        metric_sums["pos_count"] += pos_f.sum()
        metric_sums["neg_sum"] += (rho_f * neg_f).sum()
        metric_sums["neg_count"] += neg_f.sum()

        seq_len = rho.size(1)
        midpoint = max(seq_len // 2, 1)
        positions = torch.arange(seq_len, device=rho.device).unsqueeze(0)
        early = valid & positions.lt(midpoint)
        late = valid & positions.ge(midpoint)
        early_f = early.to(rho_f.dtype)
        late_f = late.to(rho_f.dtype)
        metric_sums["early_sum"] += (rho_f * early_f).sum()
        metric_sums["early_count"] += early_f.sum()
        metric_sums["late_sum"] += (rho_f * late_f).sum()
        metric_sums["late_count"] += late_f.sum()

        group = max(int(rollout_n), 1)
        for start in range(0, rho_f.size(0), group):
            end = min(start + group, rho_f.size(0))
            g_valid = valid[start:end]
            if not g_valid.any():
                continue
            g_rho = rho_f[start:end]
            g_pos = pos[start:end]
            g_neg = neg[start:end]
            g_vals = g_rho.masked_select(g_valid)
            if g_vals.numel() == 0:
                continue
            std = g_vals.std(unbiased=False).clamp_min(float(eps))
            gamma = rho_f.new_zeros(())
            if g_pos.any() and g_neg.any():
                pos_vals = g_rho.masked_select(g_pos)
                neg_vals = g_rho.masked_select(g_neg)
                pos_mean = pos_vals.mean()
                neg_mean = neg_vals.mean()
                diff = pos_mean - neg_mean
                pos_centered = pos_vals - pos_mean
                neg_centered = neg_vals - neg_mean
                pooled_count = (pos_vals.numel() + neg_vals.numel())
                pooled_var = ((pos_centered * pos_centered).sum() + (neg_centered * neg_centered).sum()) / max(pooled_count, 1)
                pooled_var = pooled_var.clamp_min(float(eps))
                metric_sums["pg_diff_sum"] += diff
                metric_sums["pg_fisher_sum"] += diff * diff / pooled_var
                metric_sums["pg_count"] += 1.0
                gamma = (diff / std).clamp(-1.0, 1.0)

            metric_sums["gamma_sum"] += gamma
            metric_sums["gamma_abs_sum"] += gamma.abs()
            metric_sums["gamma_count"] += 1.0

            med = g_vals.median()
            mad = (g_vals - med).abs().median()
            if mad > float(eps) and torch.isfinite(mad):
                z = (g_rho - med) / (mad + float(eps))
                raw = float(kappa) * gamma * z
                metric_sums["clip_sum"] += (raw.abs() > float(clip_c)).to(torch.float32).masked_select(g_valid).sum()
                metric_sums["clip_count"] += g_valid.to(torch.float32).sum()

        return metric_sums


def finalize_rho_metrics(metric_sums: dict[str, torch.Tensor] | None, eps: float = 1.0e-8) -> dict[str, float]:
    if not metric_sums:
        return {}

    count = metric_sums["count"].clamp_min(1.0)
    mean = metric_sums["sum"] / count
    var = (metric_sums["sumsq"] / count - mean * mean).clamp_min(float(eps))
    pos_mean = metric_sums["pos_sum"] / metric_sums["pos_count"].clamp_min(1.0)
    neg_mean = metric_sums["neg_sum"] / metric_sums["neg_count"].clamp_min(1.0)
    diff = pos_mean - neg_mean
    early_mean = metric_sums["early_sum"] / metric_sums["early_count"].clamp_min(1.0)
    late_mean = metric_sums["late_sum"] / metric_sums["late_count"].clamp_min(1.0)
    pg_count = metric_sums.get("pg_count", count.new_zeros(())).clamp_min(1.0)
    gamma_count = metric_sums.get("gamma_count", count.new_zeros(())).clamp_min(1.0)
    clip_count = metric_sums.get("clip_count", count.new_zeros(())).clamp_min(1.0)

    return {
        "ovc/rho_pos_minus_neg": float(diff.detach().item()),
        "ovc/rho_fisher": float((diff * diff / var).detach().item()),
        "ovc/rho_fisher_pg": float((metric_sums.get("pg_fisher_sum", count.new_zeros(())) / pg_count).detach().item()),
        "ovc/rho_pos_minus_neg_pg": float((metric_sums.get("pg_diff_sum", count.new_zeros(())) / pg_count).detach().item()),
        "ovc/rho_mean": float(mean.detach().item()),
        "ovc/rho_late_minus_early": float((late_mean - early_mean).detach().item()),
        "ovc/gamma_q_mean": float((metric_sums.get("gamma_sum", count.new_zeros(())) / gamma_count).detach().item()),
        "ovc/gamma_q_abs_mean": float((metric_sums.get("gamma_abs_sum", count.new_zeros(())) / gamma_count).detach().item()),
        "ovc/gamma_q_hist": float((metric_sums.get("gamma_sum", count.new_zeros(())) / gamma_count).detach().item()),
        "ovc/m_clip_frac": float((metric_sums.get("clip_sum", count.new_zeros(())) / clip_count).detach().item()),
    }


def blind_rho_scale(
    rho: torch.Tensor,
    response_token_mask: torch.Tensor,
    has_image: torch.Tensor,
    *,
    scale: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    with torch.no_grad():
        rho_f = rho.detach().to(torch.float32)
        valid = response_token_mask.to(device=rho.device, dtype=torch.bool)
        inf = torch.full_like(rho_f, float("inf"))
        neg_inf = torch.full_like(rho_f, -float("inf"))
        row_min = torch.where(valid, rho_f, inf).min(dim=1).values
        row_max = torch.where(valid, rho_f, neg_inf).max(dim=1).values
        valid_counts = valid.sum(dim=1).clamp_min(1).to(rho_f.dtype)
        span = row_max - row_min
        norm = torch.where(span.unsqueeze(1).gt(float(eps)), (rho_f - row_min.unsqueeze(1)) / span.unsqueeze(1), 0.5)
        norm = torch.where(valid, norm, torch.zeros_like(norm))
        centered = norm - (norm * valid.to(rho_f.dtype)).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)
        factor = (1.0 + float(scale) * centered).clamp_min(0.0)
        factor = torch.where(has_image.to(device=rho.device, dtype=torch.bool).unsqueeze(1), factor, torch.ones_like(factor))
        return factor.to(dtype=rho.dtype)
