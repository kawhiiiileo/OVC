"""Math77K training reward: 0.9 answer accuracy + 0.1 response format."""

import re

from mathruler.grader import extract_boxed_content, grade_answer


_FULL_THINK_BOXED_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
_PROMPT_OPENED_THINK_BOXED_RE = re.compile(r".*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def _format_reward(solution_str: str) -> float:
    if re.fullmatch(_FULL_THINK_BOXED_RE, solution_str):
        return 1.0
    if re.fullmatch(_PROMPT_OPENED_THINK_BOXED_RE, solution_str):
        return 1.0
    return 0.0


def _accuracy_reward(solution_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    answer = extract_boxed_content(solution_str) if use_boxed else solution_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    data_source=None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info=None,
    use_boxed: bool = True,
    format_score: float = 0.1,
    **kwargs,
):
    acc = _accuracy_reward(solution_str, ground_truth, use_boxed=use_boxed)
    fmt = _format_reward(solution_str)
    score = (1.0 - format_score) * acc + format_score * fmt
    return {
        "score": score,
        "acc_reward": acc,
        "format_reward": fmt,
    }
