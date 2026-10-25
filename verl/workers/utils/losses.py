# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.delta_visual_grounding import (
    append_rho_coordinate,
    compute_visual_grounding_rho,
    denominator_with_optional_rho_mask,
    finalize_rho_metrics,
    update_rho_metric_sums,
    weighted_sum_with_optional_rho_mask,
    weighted_sumsq_with_optional_rho_mask,
)


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _nested_to_padded(tensor: torch.Tensor, padding):
    if isinstance(tensor, torch.Tensor) and tensor.is_nested:
        return torch.nested.to_padded_tensor(tensor, padding)
    return tensor


def _visual_grounding_gate(config: ActorConfig, model_output, data: TensorDict, advantages, response_mask):
    cfg = config.get("visual_grounding", None)
    if not bool(_cfg_get(cfg, "enable", False)):
        return advantages, {}
    hidden_nested = model_output.get("delta_hidden_states", None)
    if hidden_nested is None:
        return advantages, {"ovc/missing_hidden_states": 1.0}
    if "input_ids" not in data or "prompts" not in data:
        return advantages, {"ovc/missing_input_ids": 1.0}

    signal_version = str(_cfg_get(cfg, "signal_version", "v2") or "v2").lower()
    v2_cfg = _cfg_get(cfg, "v2", {}) or {}
    whiten = str(_cfg_get(v2_cfg, "whiten", "diag") or "diag").lower()
    text_baseline = bool(_cfg_get(v2_cfg, "text_baseline", True))
    subspace_rank = int(_cfg_get(v2_cfg, "subspace_rank", 0) or 0)
    sigma_eps = max(float(_cfg_get(v2_cfg, "sigma_eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    kappa_default = 2.0 if signal_version == "v2" else 1.0
    kappa = float(_cfg_get(cfg, "kappa", kappa_default) or kappa_default)
    clip_c = float(_cfg_get(cfg, "clip_c", 0.2) or 0.2)
    eps = max(float(_cfg_get(cfg, "eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    cos_eps = max(float(_cfg_get(cfg, "cos_eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    image_token_ids = _cfg_get(cfg, "image_token_ids", [151655, 151656])
    if isinstance(image_token_ids, str):
        image_token_ids = [int(x.strip()) for x in image_token_ids.strip("[]").split(",") if x.strip()]
    if isinstance(image_token_ids, int):
        image_token_ids = [image_token_ids]

    full_hidden = _nested_to_padded(hidden_nested, 0.0).detach()
    response_hidden = no_padding_2_padding(hidden_nested, data).detach()
    input_ids = _nested_to_padded(data["input_ids"], 0)
    device = full_hidden.device
    input_ids = input_ids.to(device=device)
    B, S = input_ids.shape
    seq_lens = data["input_ids"].offsets().diff().to(device=device) if data["input_ids"].is_nested else data["attention_mask"].sum(dim=1).to(device=device)
    prompt_ids = data["prompts"]
    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff().to(device=device)
    else:
        prompt_lens = prompt_ids.ne(0).sum(dim=1).to(device=device)
    positions = torch.arange(S, device=device).unsqueeze(0)
    full_mask = positions < seq_lens.unsqueeze(1)
    prompt_mask = positions < prompt_lens.unsqueeze(1)
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for token_id in image_token_ids:
        image_mask |= input_ids.eq(int(token_id))
    image_mask &= full_mask
    text_mask = full_mask & prompt_mask & ~image_mask

    resp_mask_bool = response_mask.to(device=device, dtype=torch.bool)
    mask_f = resp_mask_bool.to(advantages.dtype)
    seq_adv = (advantages * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    rho, image_response_mask, has_image = compute_visual_grounding_rho(
        full_hidden,
        response_hidden,
        image_mask,
        resp_mask_bool,
        text_token_mask=text_mask,
        advantages=seq_adv,
        rollout_n=int(config.rollout_n or 1),
        signal_version=signal_version,
        whiten=whiten,
        text_baseline=text_baseline,
        subspace_rank=subspace_rank,
        sigma_eps=sigma_eps,
        normalize_rho=False,
        cos_eps=cos_eps,
    )

    scale = torch.ones_like(advantages, dtype=torch.float32, device=device)
    rho_f = rho.to(torch.float32)
    valid = image_response_mask & resp_mask_bool
    group = max(int(config.rollout_n or 1), 1)
    clip_sum = rho_f.new_zeros(())
    clip_count = rho_f.new_zeros(())
    gamma_sum = rho_f.new_zeros(())
    gamma_count = rho_f.new_zeros(())
    for start in range(0, rho_f.size(0), group):
        end = min(start + group, rho_f.size(0))
        g_valid = valid[start:end]
        if not g_valid.any():
            continue
        g_rho = rho_f[start:end]
        g_adv = seq_adv[start:end]
        pos = g_valid & g_adv.gt(0).unsqueeze(1)
        neg = g_valid & g_adv.lt(0).unsqueeze(1)
        gamma = rho_f.new_zeros(())
        if pos.any() and neg.any():
            vals = g_rho.masked_select(g_valid)
            std = vals.std(unbiased=False).clamp_min(eps)
            gamma = ((g_rho.masked_select(pos).mean() - g_rho.masked_select(neg).mean()) / std).clamp(-1.0, 1.0)
            med = vals.median()
            mad = (vals - med).abs().median()
            if mad > eps and torch.isfinite(mad):
                raw = kappa * gamma * ((g_rho - med) / (mad + eps))
                clipped = raw.clamp(-clip_c, clip_c)
                scale[start:end] = torch.where(g_valid, 1.0 + clipped, scale[start:end])
                clip_sum += (raw.abs() > clip_c).to(torch.float32).masked_select(g_valid).sum()
                clip_count += g_valid.to(torch.float32).sum()
        gamma_sum += gamma
        gamma_count += 1.0

    metrics_sums = update_rho_metric_sums(
        None,
        rho,
        image_response_mask,
        seq_adv,
        resp_mask_bool,
        rollout_n=int(config.rollout_n or 1),
        kappa=kappa,
        clip_c=clip_c,
        eps=eps,
    )
    metrics = finalize_rho_metrics(metrics_sums, eps=eps)
    metrics["ovc/enabled"] = 1.0
    metrics["ovc/signal_version_v2"] = 1.0 if signal_version == "v2" else 0.0
    metrics["ovc/v2_whiten_diag"] = 1.0 if whiten == "diag" else 0.0
    metrics["ovc/v2_text_baseline"] = 1.0 if text_baseline else 0.0
    metrics["ovc/gate_scale_mean"] = float(scale.masked_select(resp_mask_bool).mean().detach().item()) if resp_mask_bool.any() else 1.0
    metrics["ovc/has_image_frac"] = float(has_image.to(torch.float32).mean().detach().item())
    if clip_count > 0:
        metrics["ovc/m_clip_frac"] = float((clip_sum / clip_count.clamp_min(1.0)).detach().item())
    if gamma_count > 0:
        metrics["ovc/gamma_q_mean"] = float((gamma_sum / gamma_count.clamp_min(1.0)).detach().item())
    return advantages * scale.to(dtype=advantages.dtype), metrics



def _as_float(value, default):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _align_time(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if tensor.size(1) == target_len:
        return tensor
    if tensor.size(1) > target_len:
        return tensor[:, -target_len:]
    pad_shape = list(tensor.shape)
    pad_shape[1] = target_len - tensor.size(1)
    pad = tensor.new_zeros(pad_shape)
    return torch.cat([pad, tensor], dim=1)


def _sequence_advantage(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.dim() <= 1:
        return advantages.detach().to(torch.float32)
    return advantages.detach().to(torch.float32)[:, 0]


def _prompt_lengths_from_data(data: TensorDict, input_ids: torch.Tensor, response_len: int) -> torch.Tensor:
    device = input_ids.device
    if "prompts" in data:
        prompts = data["prompts"]
        if isinstance(prompts, torch.Tensor) and prompts.is_nested:
            return prompts.offsets().diff().to(device=device)
        if isinstance(prompts, torch.Tensor):
            return prompts.ne(0).sum(dim=1).to(device=device)
    if isinstance(data["input_ids"], torch.Tensor) and data["input_ids"].is_nested:
        seq_lens = data["input_ids"].offsets().diff().to(device=device)
    elif "attention_mask" in data:
        seq_lens = data["attention_mask"].sum(dim=1).to(device=device)
    else:
        seq_lens = torch.full((input_ids.size(0),), input_ids.size(1), device=device, dtype=torch.long)
    return (seq_lens - int(response_len)).clamp_min(0)



def _delta_impl_info(config: ActorConfig):
    impl = config.get("impl", None)
    if impl in (None, ""):
        return False, False
    version_key = str(impl).replace("-", "_").lower()
    if version_key in {"none", "disable", "disabled", "baseline", "dapo", "gspo"}:
        return False, False
    if version_key == "normal":
        return True, False
    if version_key in {"mahalanobis", "maha", "mahalanobis_memory_efficient", "memory_efficient_mahalanobis", "maha_memory_efficient", "memory_efficient_maha"}:
        return True, True
    raise ValueError(f"Unsupported DelTA impl: {impl}")


def delta_global_stats_enabled(config: ActorConfig) -> bool:
    enabled, _ = _delta_impl_info(config)
    return bool(enabled)


def delta_global_stats_reset(config: ActorConfig) -> None:
    object.__setattr__(config, "_delta_global_stats", None)


def _delta_build_token_proxy(config: ActorConfig, model_output, data: TensorDict, log_prob, response_mask):
    hidden_nested = model_output.get("delta_hidden_states", None)
    if hidden_nested is None:
        return None
    with torch.no_grad():
        response_mask = response_mask.to(dtype=torch.bool)
        mask_f = response_mask.to(dtype=torch.float32)
        advantages = data.select("advantages").to_padded_tensor()["advantages"]
        seq_adv = _sequence_advantage(advantages)
        response_hidden = no_padding_2_padding(hidden_nested, data).detach().to(torch.float32)
        response_hidden = _align_time(response_hidden, response_mask.size(1))
        log_prob_detached = _align_time(log_prob.detach().to(torch.float32), response_mask.size(1))
        probs = torch.exp(log_prob_detached).clamp(0.0, 1.0)
        v = (1.0 - probs).unsqueeze(-1) * response_hidden
        ovc_batch, metrics = _build_visual_grounding_batch(config, model_output, data, response_hidden, response_mask, seq_adv)
        ovc_cfg = config.get("visual_grounding", None)
        ovc_discriminative = ovc_batch is not None and str(_cfg_get(ovc_cfg, "gating", "discriminative") or "discriminative").lower() in {"discriminative", "signed"}
        ovc_stat_mask = ovc_batch["image_response_mask"] if ovc_discriminative else None
        if ovc_discriminative:
            v = append_rho_coordinate(v, ovc_batch["rho"])
        return {
            "v": v.detach(),
            "seq_adv": seq_adv.detach(),
            "response_mask": response_mask.detach(),
            "mask_f": mask_f.detach(),
            "ovc_stat_mask": ovc_stat_mask.detach() if ovc_stat_mask is not None else None,
            "metrics": metrics,
        }



def _delta_margin_terms(v, mu_pos, mu_neg, diag):
    diff_pos = v - mu_pos.view(1, 1, -1)
    diff_neg = v - mu_neg.view(1, 1, -1)
    dist_pos = ((diff_pos * diff_pos) / diag.view(1, 1, -1)).sum(dim=-1)
    dist_neg = ((diff_neg * diff_neg) / diag.view(1, 1, -1)).sum(dim=-1)
    return dist_neg - dist_pos, dist_pos - dist_neg


def _delta_accum_margin(stats, margin_pos, margin_neg, valid_pos, valid_neg):
    for prefix, margin, valid in (("pos", margin_pos, valid_pos), ("neg", margin_neg, valid_neg)):
        valid_f = valid.to(margin.dtype)
        stats[f"m_{prefix}_sum"] += (margin * valid_f).sum().detach().to(torch.float32)
        stats[f"m_{prefix}_sumsq"] += (margin * margin * valid_f).sum().detach().to(torch.float32)
        stats[f"m_{prefix}_count"] += valid_f.sum().detach().to(torch.float32)


def _delta_finalize_gamma(stats, eps):
    for prefix in ("pos", "neg"):
        count = stats[f"m_{prefix}_count"].clamp_min(1.0)
        mean = stats[f"m_{prefix}_sum"] / count
        var = (stats[f"m_{prefix}_sumsq"] / count - mean * mean).clamp_min(eps)
        stats[f"gamma_{prefix}"] = torch.sqrt(var).detach()


def _delta_new_margin_stats(device, dtype=torch.float32):
    return {
        "m_pos_sum": torch.zeros((), device=device, dtype=dtype),
        "m_pos_sumsq": torch.zeros((), device=device, dtype=dtype),
        "m_pos_count": torch.zeros((), device=device, dtype=dtype),
        "m_neg_sum": torch.zeros((), device=device, dtype=dtype),
        "m_neg_sumsq": torch.zeros((), device=device, dtype=dtype),
        "m_neg_count": torch.zeros((), device=device, dtype=dtype),
    }


def delta_global_stats_set_stage(config: ActorConfig, stage: str) -> None:
    stats = config.get("_delta_global_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    stats["stage"] = stage
    object.__setattr__(config, "_delta_global_stats", stats)


def delta_global_stats_accumulate(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    enabled, use_mahalanobis = _delta_impl_info(config)
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    zero = log_prob.sum() * 0.0
    if not enabled:
        return zero, {}
    response_mask = data.select("response_mask").to_padded_tensor()["response_mask"].to(bool)
    proxy = _delta_build_token_proxy(config, model_output, data, log_prob, response_mask)
    if proxy is None:
        object.__setattr__(config, "_delta_global_stats", {"missing_hidden_states": True})
        return zero, {"delta/missing_hidden_states": 1.0}

    v = proxy["v"]
    seq_adv = proxy["seq_adv"]
    mask_f = proxy["mask_f"]
    ovc_stat_mask = proxy["ovc_stat_mask"]
    pos_seq = seq_adv.gt(0).unsqueeze(1)
    neg_seq = seq_adv.lt(0).unsqueeze(1)
    valid_pos = pos_seq & response_mask
    valid_neg = neg_seq & response_mask
    adv_pos = seq_adv.clamp_min(0.0).unsqueeze(1) * mask_f
    adv_neg = (-seq_adv).clamp_min(0.0).unsqueeze(1) * mask_f

    stats = config.get("_delta_global_stats", None)
    if not isinstance(stats, dict) or stats.get("missing_hidden_states", False):
        stats = {"stage": "base"}
    stage = stats.get("stage", "base")
    eps = 1.0e-8

    if stage == "base":
        if "sum_pos" not in stats:
            stats.update({
                "sum_pos": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "sum_neg": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "sumsq_pos": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "sumsq_neg": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "den_pos": torch.zeros(v.size(-1) if ovc_stat_mask is not None else 1, device=v.device, dtype=torch.float32),
                "den_neg": torch.zeros(v.size(-1) if ovc_stat_mask is not None else 1, device=v.device, dtype=torch.float32),
                "has_ovc_coord": bool(ovc_stat_mask is not None),
                "use_mahalanobis": bool(use_mahalanobis),
            })
        stats["sum_pos"] += weighted_sum_with_optional_rho_mask(adv_pos, v, ovc_stat_mask).detach().to(torch.float32)
        stats["sum_neg"] += weighted_sum_with_optional_rho_mask(adv_neg, v, ovc_stat_mask).detach().to(torch.float32)
        stats["sumsq_pos"] += weighted_sumsq_with_optional_rho_mask(adv_pos, v, ovc_stat_mask).detach().to(torch.float32)
        stats["sumsq_neg"] += weighted_sumsq_with_optional_rho_mask(adv_neg, v, ovc_stat_mask).detach().to(torch.float32)
        stats["den_pos"] += denominator_with_optional_rho_mask(adv_pos, v, ovc_stat_mask).detach().to(torch.float32)
        stats["den_neg"] += denominator_with_optional_rho_mask(adv_neg, v, ovc_stat_mask).detach().to(torch.float32)
    elif stage == "margin":
        if "m_pos_sum" not in stats:
            stats.update(_delta_new_margin_stats(v.device))
        margin_pos, margin_neg = _delta_margin_terms(v, stats["mu_pos"].to(v), stats["mu_neg"].to(v), stats["diag"].to(v))
        _delta_accum_margin(stats, margin_pos, margin_neg, valid_pos, valid_neg)
    elif stage == "refine":
        margin_pos, margin_neg = _delta_margin_terms(v, stats["mu_pos_star"].to(v), stats["mu_neg_star"].to(v), stats["diag"].to(v))
        gamma_pos = stats["gamma_pos"].to(v).clamp_min(eps)
        gamma_neg = stats["gamma_neg"].to(v).clamp_min(eps)
        raw_pos = torch.sigmoid(margin_pos / gamma_pos) * adv_pos * valid_pos.to(v.dtype)
        raw_neg = torch.sigmoid(margin_neg / gamma_neg) * adv_neg * valid_neg.to(v.dtype)
        if "num_pos" not in stats:
            stats.update({
                "num_pos": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "num_neg": torch.zeros(v.size(-1), device=v.device, dtype=torch.float32),
                "den_pos_star": torch.zeros(v.size(-1) if ovc_stat_mask is not None else 1, device=v.device, dtype=torch.float32),
                "den_neg_star": torch.zeros(v.size(-1) if ovc_stat_mask is not None else 1, device=v.device, dtype=torch.float32),
            })
            stats.update(_delta_new_margin_stats(v.device))
        stats["num_pos"] += weighted_sum_with_optional_rho_mask(raw_pos, v, ovc_stat_mask).detach().to(torch.float32)
        stats["num_neg"] += weighted_sum_with_optional_rho_mask(raw_neg, v, ovc_stat_mask).detach().to(torch.float32)
        stats["den_pos_star"] += denominator_with_optional_rho_mask(raw_pos, v, ovc_stat_mask).detach().to(torch.float32)
        stats["den_neg_star"] += denominator_with_optional_rho_mask(raw_neg, v, ovc_stat_mask).detach().to(torch.float32)
        _delta_accum_margin(stats, margin_pos, margin_neg, valid_pos, valid_neg)
    elif stage == "norm":
        margin_pos, margin_neg = _delta_margin_terms(v, stats["mu_pos_star"].to(v), stats["mu_neg_star"].to(v), stats["diag"].to(v))
        final_lo = _as_float(config.get("lam_min", 0.8), 0.8)
        final_hi = _as_float(config.get("lam_max", 1.2), 1.2)
        w_pos = final_lo + (final_hi - final_lo) * torch.sigmoid(margin_pos / stats["gamma_pos"].to(v).clamp_min(eps))
        w_neg = final_lo + (final_hi - final_lo) * torch.sigmoid(margin_neg / stats["gamma_neg"].to(v).clamp_min(eps))
        weight = (w_pos * pos_seq.to(v.dtype) + w_neg * neg_seq.to(v.dtype) + final_lo * (~(pos_seq | neg_seq)).to(v.dtype)) * mask_f
        if "tk_sum" not in stats:
            stats["tk_sum"] = torch.zeros((), device=v.device, dtype=torch.float32)
            stats["wht_sum"] = torch.zeros((), device=v.device, dtype=torch.float32)
        stats["tk_sum"] += mask_f.sum().detach().to(torch.float32)
        stats["wht_sum"] += weight.sum().detach().to(torch.float32)
    else:
        raise ValueError(f"Unknown DelTA global stats stage: {stage}")

    object.__setattr__(config, "_delta_global_stats", stats)
    return zero, {}


def delta_global_stats_finalize(config: ActorConfig, dp_group=None) -> None:
    stats = config.get("_delta_global_stats", None)
    if not isinstance(stats, dict) or stats.get("missing_hidden_states", False):
        return
    stage = stats.get("stage", "base")
    eps = 1.0e-8
    if stage == "base":
        for key in ("sum_pos", "sum_neg", "sumsq_pos", "sumsq_neg", "den_pos", "den_neg"):
            if dp_group is not None:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM, group=dp_group)
        den_pos = stats["den_pos"].clamp_min(eps)
        den_neg = stats["den_neg"].clamp_min(eps)
        mu_pos = stats["sum_pos"] / den_pos
        mu_neg = stats["sum_neg"] / den_neg
        sig_pos = (stats["sumsq_pos"] / den_pos - mu_pos * mu_pos).clamp_min(0.0)
        sig_neg = (stats["sumsq_neg"] / den_neg - mu_neg * mu_neg).clamp_min(0.0)
        diag = _delta_metric_diag(config, sig_pos, sig_neg, bool(stats.get("use_mahalanobis", False)), eps, bool(stats.get("has_ovc_coord", False)))
        stats.update({"mu_pos": mu_pos.detach(), "mu_neg": mu_neg.detach(), "diag": diag.detach()})
    elif stage == "margin":
        for key in ("m_pos_sum", "m_pos_sumsq", "m_pos_count", "m_neg_sum", "m_neg_sumsq", "m_neg_count"):
            if dp_group is not None:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM, group=dp_group)
        _delta_finalize_gamma(stats, eps)
        stats["mu_pos_star"] = stats["mu_pos"].detach().clone()
        stats["mu_neg_star"] = stats["mu_neg"].detach().clone()
    elif stage == "refine":
        for key in ("num_pos", "num_neg", "den_pos_star", "den_neg_star", "m_pos_sum", "m_pos_sumsq", "m_pos_count", "m_neg_sum", "m_neg_sumsq", "m_neg_count"):
            if dp_group is not None:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM, group=dp_group)
        stats["mu_pos_star"] = (stats["num_pos"] / stats["den_pos_star"].clamp_min(eps)).detach()
        stats["mu_neg_star"] = (stats["num_neg"] / stats["den_neg_star"].clamp_min(eps)).detach()
        _delta_finalize_gamma(stats, eps)
        for key in ("num_pos", "num_neg", "den_pos_star", "den_neg_star", "m_pos_sum", "m_pos_sumsq", "m_pos_count", "m_neg_sum", "m_neg_sumsq", "m_neg_count"):
            stats.pop(key, None)
    elif stage == "norm":
        for key in ("tk_sum", "wht_sum"):
            if dp_group is not None:
                torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM, group=dp_group)
        stats["gy_norm"] = (stats["tk_sum"] / stats["wht_sum"].clamp_min(eps)).detach()
        stats["finalized"] = True
    object.__setattr__(config, "_delta_global_stats", stats)


def delta_global_stats_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    return delta_global_stats_accumulate(config, model_output, data, dp_group=dp_group)

def _delta_metric_diag(config: ActorConfig, sig_pos: torch.Tensor, sig_neg: torch.Tensor, use_mahalanobis: bool, eps: float, ovc_discriminative: bool) -> torch.Tensor:
    from ovc_release.credit.calibration import diagonal_metric
    alpha = config.get("maha_alpha", None)
    alpha = None if alpha in (None, "") else min(max(float(alpha), 0.0), 1.0)
    return diagonal_metric(
        sig_pos, sig_neg, epsilon=eps,
        shrink=min(max(_as_float(config.get("maha_shrink", 0.1), 0.1), 0.0), 1.0),
        min_var_scale=max(_as_float(config.get("maha_min_var_scale", 1.0e-3), 1.0e-3), 0.0),
        alpha=alpha, power=max(_as_float(config.get("maha_power", 1.0), 1.0), 0.0),
        exclude_visual=ovc_discriminative, enabled=use_mahalanobis, floor_unit=True,
    )


def _build_visual_grounding_batch(config: ActorConfig, model_output, data: TensorDict, response_hidden, response_mask, seq_adv):
    cfg = config.get("visual_grounding", None)
    if not bool(_cfg_get(cfg, "enable", False)):
        return None, {}
    hidden_nested = model_output.get("delta_hidden_states", None)
    if hidden_nested is None:
        return None, {"ovc/missing_hidden_states": 1.0}
    if "input_ids" not in data:
        return None, {"ovc/missing_input_ids": 1.0}

    signal_version = str(_cfg_get(cfg, "signal_version", "v2") or "v2").lower()
    v2_cfg = _cfg_get(cfg, "v2", {}) or {}
    whiten = str(_cfg_get(v2_cfg, "whiten", "diag") or "diag").lower()
    text_baseline = bool(_cfg_get(v2_cfg, "text_baseline", True))
    subspace_rank = int(_cfg_get(v2_cfg, "subspace_rank", 0) or 0)
    sigma_eps = max(float(_cfg_get(v2_cfg, "sigma_eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    cos_eps = max(float(_cfg_get(cfg, "cos_eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    normalize_rho = bool(_cfg_get(cfg, "normalize_rho", False))
    image_token_ids = _cfg_get(cfg, "image_token_ids", [151655, 151656])
    if isinstance(image_token_ids, str):
        image_token_ids = [int(x.strip()) for x in image_token_ids.strip("[]").split(",") if x.strip()]
    if isinstance(image_token_ids, int):
        image_token_ids = [image_token_ids]

    full_hidden = _nested_to_padded(hidden_nested, 0.0).detach()
    input_ids = _nested_to_padded(data["input_ids"], 0).to(device=full_hidden.device)
    S = min(full_hidden.size(1), input_ids.size(1))
    full_hidden = full_hidden[:, :S]
    input_ids = input_ids[:, :S]
    if isinstance(data["input_ids"], torch.Tensor) and data["input_ids"].is_nested:
        seq_lens = data["input_ids"].offsets().diff().to(device=full_hidden.device).clamp_max(S)
    elif "attention_mask" in data:
        seq_lens = data["attention_mask"].sum(dim=1).to(device=full_hidden.device).clamp_max(S)
    else:
        seq_lens = torch.full((input_ids.size(0),), S, device=full_hidden.device, dtype=torch.long)
    prompt_lens = _prompt_lengths_from_data(data, input_ids, response_mask.size(1)).clamp_max(S)
    positions = torch.arange(S, device=full_hidden.device).unsqueeze(0)
    full_mask = positions < seq_lens.unsqueeze(1)
    prompt_mask = positions < prompt_lens.unsqueeze(1)
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=full_hidden.device)
    for token_id in image_token_ids:
        image_mask |= input_ids.eq(int(token_id))
    image_mask &= full_mask
    text_mask = full_mask & prompt_mask & ~image_mask

    rho, image_response_mask, has_image = compute_visual_grounding_rho(
        full_hidden,
        response_hidden,
        image_mask,
        response_mask,
        text_token_mask=text_mask,
        advantages=seq_adv,
        rollout_n=int(config.rollout_n or 1),
        signal_version=signal_version,
        whiten=whiten,
        text_baseline=text_baseline,
        subspace_rank=subspace_rank,
        sigma_eps=sigma_eps,
        normalize_rho=normalize_rho,
        cos_eps=cos_eps,
    )
    eps = max(float(_cfg_get(cfg, "eps", 1.0e-6) or 1.0e-6), 1.0e-12)
    kappa = float(_cfg_get(cfg, "kappa", 2.0 if signal_version == "v2" else 1.0) or 1.0)
    clip_c = float(_cfg_get(cfg, "clip_c", 0.2) or 0.2)
    sums = update_rho_metric_sums(None, rho, image_response_mask, seq_adv, response_mask, rollout_n=int(config.rollout_n or 1), kappa=kappa, clip_c=clip_c, eps=eps)
    metrics = finalize_rho_metrics(sums, eps=eps)
    metrics.update({
        "ovc/enabled": 1.0,
        "ovc/missing_hidden_states": 0.0,
        "ovc/signal_version_v2": 1.0 if signal_version == "v2" else 0.0,
        "ovc/v2_whiten_diag": 1.0 if whiten == "diag" else 0.0,
        "ovc/v2_text_baseline": 1.0 if text_baseline else 0.0,
        "ovc/has_image_frac": float(has_image.to(torch.float32).mean().detach().item()),
    })
    return {"rho": rho, "image_response_mask": image_response_mask, "has_image": has_image}, metrics


def _delta_advantage_reweight(config: ActorConfig, model_output, data: TensorDict, log_prob, advantages, response_mask):
    impl = config.get("impl", None)
    if impl in (None, ""):
        return advantages, {}
    version_key = str(impl).replace("-", "_").lower()
    if version_key in {"none", "disable", "disabled", "baseline", "dapo", "gspo"}:
        return advantages, {}
    if version_key == "normal":
        use_mahalanobis = False
    elif version_key in {"mahalanobis", "maha", "mahalanobis_memory_efficient", "memory_efficient_mahalanobis", "maha_memory_efficient", "memory_efficient_maha"}:
        use_mahalanobis = True
    else:
        raise ValueError(f"Unsupported DelTA impl: {impl}")

    hidden_nested = model_output.get("delta_hidden_states", None)
    if hidden_nested is None:
        return advantages, {"delta/enabled": 1.0, "delta/missing_hidden_states": 1.0}

    with torch.no_grad():
        eps = 1.0e-8
        response_mask = response_mask.to(dtype=torch.bool)
        mask_f = response_mask.to(dtype=torch.float32)
        seq_adv = _sequence_advantage(advantages)
        response_hidden = no_padding_2_padding(hidden_nested, data).detach().to(torch.float32)
        response_hidden = _align_time(response_hidden, response_mask.size(1))
        log_prob_detached = _align_time(log_prob.detach().to(torch.float32), response_mask.size(1))
        probs = torch.exp(log_prob_detached).clamp(0.0, 1.0)
        v = (1.0 - probs).unsqueeze(-1) * response_hidden

        ovc_batch, metrics = _build_visual_grounding_batch(config, model_output, data, response_hidden, response_mask, seq_adv)
        ovc_cfg = config.get("visual_grounding", None)
        ovc_discriminative = ovc_batch is not None and str(_cfg_get(ovc_cfg, "gating", "discriminative") or "discriminative").lower() in {"discriminative", "signed"}
        ovc_stat_mask = ovc_batch["image_response_mask"] if ovc_discriminative else None
        if ovc_discriminative:
            v = append_rho_coordinate(v, ovc_batch["rho"])

        pos_seq = seq_adv.gt(0).unsqueeze(1)
        neg_seq = seq_adv.lt(0).unsqueeze(1)
        valid_pos = response_mask & pos_seq
        valid_neg = response_mask & neg_seq

        global_stats = config.get("_delta_global_stats", None)
        use_global_stats = isinstance(global_stats, dict) and global_stats.get("finalized", False)
        if use_global_stats:
            mu_pos = global_stats["mu_pos"].to(device=v.device, dtype=v.dtype)
            mu_neg = global_stats["mu_neg"].to(device=v.device, dtype=v.dtype)
            diag = global_stats["diag"].to(device=v.device, dtype=v.dtype)
            metrics["delta/global_stats"] = 1.0
        else:
            if not valid_pos.any() or not valid_neg.any():
                metrics.update({"delta/enabled": 1.0, "delta/skipped_no_pos_neg": 1.0, "delta/global_stats": 0.0})
                return advantages, metrics
            adv_pos = seq_adv.clamp_min(0.0).unsqueeze(1) * mask_f
            adv_neg = (-seq_adv).clamp_min(0.0).unsqueeze(1) * mask_f
            sum_pos = weighted_sum_with_optional_rho_mask(adv_pos, v, ovc_stat_mask)
            sum_neg = weighted_sum_with_optional_rho_mask(adv_neg, v, ovc_stat_mask)
            sumsq_pos = weighted_sumsq_with_optional_rho_mask(adv_pos, v, ovc_stat_mask)
            sumsq_neg = weighted_sumsq_with_optional_rho_mask(adv_neg, v, ovc_stat_mask)
            den_pos = denominator_with_optional_rho_mask(adv_pos, v, ovc_stat_mask).clamp_min(eps)
            den_neg = denominator_with_optional_rho_mask(adv_neg, v, ovc_stat_mask).clamp_min(eps)
            mu_pos = sum_pos / den_pos
            mu_neg = sum_neg / den_neg
            sig_pos = (sumsq_pos / den_pos - mu_pos * mu_pos).clamp_min(0.0)
            sig_neg = (sumsq_neg / den_neg - mu_neg * mu_neg).clamp_min(0.0)
            diag = _delta_metric_diag(config, sig_pos, sig_neg, use_mahalanobis, eps, ovc_discriminative)
            metrics["delta/global_stats"] = 0.0

        adv_pos = seq_adv.clamp_min(0.0).unsqueeze(1) * mask_f
        adv_neg = (-seq_adv).clamp_min(0.0).unsqueeze(1) * mask_f
        K = max(int(_as_float(config.get("K", 1), 1)), 0)
        if use_global_stats:
            mu_pos_star = global_stats.get("mu_pos_star", global_stats["mu_pos"]).to(device=v.device, dtype=v.dtype)
            mu_neg_star = global_stats.get("mu_neg_star", global_stats["mu_neg"]).to(device=v.device, dtype=v.dtype)
            gamma_pos = global_stats["gamma_pos"].to(device=v.device, dtype=v.dtype).clamp_min(eps)
            gamma_neg = global_stats["gamma_neg"].to(device=v.device, dtype=v.dtype).clamp_min(eps)
        else:
            mu_pos_star = mu_pos
            mu_neg_star = mu_neg
            gamma_pos = v.new_tensor(1.0)
            gamma_neg = v.new_tensor(1.0)
            pos_weight = pos_seq.to(v.dtype) * mask_f
            neg_weight = neg_seq.to(v.dtype) * mask_f
            for _ in range(K):
                diff_pos = v - mu_pos_star.view(1, 1, -1)
                diff_neg = v - mu_neg_star.view(1, 1, -1)
                dist_pos = ((diff_pos * diff_pos) / diag.view(1, 1, -1)).sum(dim=-1)
                dist_neg = ((diff_neg * diff_neg) / diag.view(1, 1, -1)).sum(dim=-1)
                margin_pos = dist_neg - dist_pos
                margin_neg = dist_pos - dist_neg
                if valid_pos.any():
                    vals = margin_pos.masked_select(valid_pos)
                    gamma_pos = vals.std(unbiased=False).clamp_min(eps) if vals.numel() > 1 else vals.new_tensor(1.0)
                if valid_neg.any():
                    vals = margin_neg.masked_select(valid_neg)
                    gamma_neg = vals.std(unbiased=False).clamp_min(eps) if vals.numel() > 1 else vals.new_tensor(1.0)
                raw_pos = torch.sigmoid(margin_pos / gamma_pos.clamp_min(eps)) * adv_pos * pos_weight
                raw_neg = torch.sigmoid(margin_neg / gamma_neg.clamp_min(eps)) * adv_neg * neg_weight
                den_pos_star = denominator_with_optional_rho_mask(raw_pos, v, ovc_stat_mask).clamp_min(eps)
                den_neg_star = denominator_with_optional_rho_mask(raw_neg, v, ovc_stat_mask).clamp_min(eps)
                mu_pos_star = weighted_sum_with_optional_rho_mask(raw_pos, v, ovc_stat_mask) / den_pos_star
                mu_neg_star = weighted_sum_with_optional_rho_mask(raw_neg, v, ovc_stat_mask) / den_neg_star

        diff_pos = v - mu_pos_star.view(1, 1, -1)
        diff_neg = v - mu_neg_star.view(1, 1, -1)
        dist_pos = ((diff_pos * diff_pos) / diag.view(1, 1, -1)).sum(dim=-1)
        dist_neg = ((diff_neg * diff_neg) / diag.view(1, 1, -1)).sum(dim=-1)
        margin_pos = dist_neg - dist_pos
        margin_neg = dist_pos - dist_neg
        final_lo = _as_float(config.get("lam_min", 0.8), 0.8)
        final_hi = _as_float(config.get("lam_max", 1.2), 1.2)
        w_pos = final_lo + (final_hi - final_lo) * torch.sigmoid(margin_pos / gamma_pos.clamp_min(eps))
        w_neg = final_lo + (final_hi - final_lo) * torch.sigmoid(margin_neg / gamma_neg.clamp_min(eps))
        weight = (w_pos * pos_seq.to(w_pos.dtype) + w_neg * neg_seq.to(w_neg.dtype) + final_lo * (~(pos_seq | neg_seq)).to(w_pos.dtype)) * mask_f
        if use_global_stats and "gy_norm" in global_stats:
            norm = global_stats["gy_norm"].to(device=weight.device, dtype=weight.dtype)
        else:
            norm = mask_f.sum() / weight.sum().clamp_min(eps)
        weight = weight * norm
        transformed = advantages * weight.to(dtype=advantages.dtype)

        valid_weight = weight.masked_select(response_mask)
        metrics.update({
            "delta/enabled": 1.0,
            "delta/missing_hidden_states": 0.0,
            "delta/impl_mahalanobis": 1.0 if use_mahalanobis else 0.0,
            "delta/K": float(K),
            "delta/weight_mean": float(valid_weight.mean().detach().item()),
            "delta/weight_min": float(valid_weight.min().detach().item()),
            "delta/weight_max": float(valid_weight.max().detach().item()),
            "delta/pos_seq_frac": float(pos_seq.to(torch.float32).mean().detach().item()),
            "delta/neg_seq_frac": float(neg_seq.to(torch.float32).mean().detach().item()),
            "delta/diag_mean": float(diag.mean().detach().item()),
        })
        return transformed, metrics

def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}
    full_data = data

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    advantages, delta_metrics = _delta_advantage_reweight(config, model_output, full_data, log_prob, advantages, response_mask)
    if not delta_metrics:
        advantages, delta_metrics = _visual_grounding_gate(config, model_output, full_data, advantages, response_mask)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    if delta_metrics:
        metrics.update(Metric.from_dict(delta_metrics, aggregation=AggregationType.MEAN))
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
