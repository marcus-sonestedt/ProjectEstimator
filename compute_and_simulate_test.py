import unittest
import numpy as np
from compute_and_simulate import bucketify, Task, TaskCompletionEstimator

class TestBucketify(unittest.TestCase):
    def test_bucketify_empty(self):
        # Test with empty string
        with self.assertRaises(ValueError):
            bucketify("")
    
    def test_bucketify_single_value(self):
        # Test with a single value
        result = bucketify("5")
        self.assertEqual(len(result), 1)  # Should create only one bucket for a single value
        
    def test_bucketify_multiple_values(self):
        # Test with multiple values
        votes = "1 2 3 4 5 6 7 8 9 10"
        result = bucketify(votes)
        self.assertEqual(sum(result.values()), 10)  # Total count should equal number of votes
        
    def test_bucketify_range(self):
        # Test if buckets cover the range of values
        votes = "1 10 10 10"
        result = bucketify(votes)
        # Check if the bucket containing 10 has a count of 3
        found = False
        for key, value in result.items():
            if "10.0" in key and value == 3:
                found = True
                break
        self.assertTrue(found, "Bucket containing 10 should have count of 3")

class TestTask(unittest.TestCase):
    def test_task_initialization(self):
        # Test task initialization with valid input
        row = [0, "Task 1", "1 2 3", "4 5 6", "7 8 9"]
        task = Task(row)
        self.assertEqual(task.name, "Task 1")
        self.assertIn('O', task.buckets)
        self.assertIn('L', task.buckets)
        self.assertIn('P', task.buckets)

class TestProbulator(unittest.TestCase):
    def setUp(self):
        self.probulator = TaskCompletionEstimator()
    
    def test_simulate(self):
        # Test simulation
        self.probulator.simulate()
        self.assertEqual(len(self.probulator.probabilities), 100)
        self.assertEqual(len(self.probulator.cumulative_probabilities), 100)
        
    def test_plot_method_exists(self):
        # Test that plot method exists and doesn't require parameters
        import inspect
        signature = inspect.signature(self.probulator.plot)
        self.assertEqual(len(signature.parameters), 0)

if __name__ == "__main__":
    unittest.main()