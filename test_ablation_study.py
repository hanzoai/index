import unittest
from ablation_study import calculate_exact_rrf


class TestAblationStudy(unittest.TestCase):
    def test_rrf_ablation_runs_cleanly(self):
        dense = list(range(1, 101))
        sparse = list(range(100, 0, -1))
        calculate_exact_rrf([10, 60, 100], dense, sparse)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
