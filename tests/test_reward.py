"""Check the actual training grader and its accuracy/format reward contract."""
import unittest

from recipe.math77k.reward import compute_score


class RewardTests(unittest.TestCase):
    def test_correct_formatted_answer_receives_full_reward(self):
        result = compute_score(solution_str=r'<think>compute</think>\boxed{2}', ground_truth='2')
        self.assertEqual(result, {'score': 1.0, 'acc_reward': 1.0, 'format_reward': 1.0})

    def test_format_does_not_make_an_incorrect_answer_accurate(self):
        result = compute_score(solution_str=r'<think>compute</think>\boxed{3}', ground_truth='2')
        self.assertEqual(result['score'], 0.1)
        self.assertEqual(result['acc_reward'], 0.0)

    def test_correct_boxed_answer_without_thinking_format(self):
        result = compute_score(solution_str=r'\boxed{2}', ground_truth='2')
        self.assertEqual(result['score'], 0.9)
        self.assertEqual(result['format_reward'], 0.0)
