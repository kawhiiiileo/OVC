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

"""RVE and credit tensor utilities shared across training backends."""
from ovc_release.credit.visual import (
    PromptGroupSigmaAccumulator,
    sigma_for_prompt_uids,
    _masked_mean,
    _cosine,
    _response_sigma,
    compute_visual_grounding_rho,
    append_rho_coordinate,
    weighted_sum_with_optional_rho_mask,
    weighted_sumsq_with_optional_rho_mask,
    denominator_with_optional_rho_mask,
    update_rho_metric_sums,
    finalize_rho_metrics,
    blind_rho_scale,
)
