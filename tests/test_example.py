# tests/test_example.py
import unittest

class TestExample(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)  # Basic addition test

    def test_subtraction(self):
        self.assertEqual(2 - 1, 1)  # Basic subtraction test

if __name__ == "__main__":
    unittest.main()
