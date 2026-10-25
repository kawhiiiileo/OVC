# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""Shared diagonal evidence metric; aggregation stays with the trainer."""
import torch


@torch.no_grad()
def diagonal_metric(
    sig_pos, sig_neg, *, epsilon, shrink=0.1, min_var_scale=1.0e-3,
    alpha=None, power=1.0, exclude_visual=False, reference=None,
    enabled=True, floor_unit=False,
):
    """Regularize within-side variation without mixing backend batch scopes."""
    reference = sig_pos if reference is None else reference
    if not enabled:
        unit = torch.ones_like(reference)
        return unit.clamp_min(epsilon) if floor_unit else unit
    diag = torch.nan_to_num(sig_pos + sig_neg, nan=0.0, posinf=0.0, neginf=0.0)
    if exclude_visual and diag.numel() > 1:
        diag_mean = diag[:-1].mean().detach().clamp_min(epsilon)
    else:
        diag_mean = diag.mean().detach().clamp_min(epsilon)
    if shrink > 0.0:
        diag = (1.0 - shrink) * diag + shrink * diag_mean
    diag = torch.clamp(diag + epsilon, min=diag_mean * min_var_scale + epsilon)
    if alpha is None:
        return diag
    norm_diag = diag / diag_mean
    norm_diag = norm_diag.clamp_min(min_var_scale + epsilon).pow(power)
    return (1.0 - alpha) * torch.ones_like(reference) + alpha * norm_diag
