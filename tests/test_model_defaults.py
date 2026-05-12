import unittest

import transcribe


class ModelDefaultsTest(unittest.TestCase):
    def test_fn_default_model_is_cohere(self):
        self.assertEqual(transcribe.DEFAULT_MODEL_MODE, "cohere")
        self.assertEqual(transcribe.MENU_MODEL_MODES[0], "cohere")


if __name__ == "__main__":
    unittest.main()
