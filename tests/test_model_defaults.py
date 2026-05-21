import unittest

import transcribe


class ModelDefaultsTest(unittest.TestCase):
    def test_fn_default_model_is_qwen_fast(self):
        self.assertEqual(transcribe.DEFAULT_MODEL_MODE, "fast")
        self.assertEqual(transcribe.MENU_MODEL_MODES[0], "fast")
        self.assertIn("4-bit", transcribe.MODEL_LABELS["fast"])


if __name__ == "__main__":
    unittest.main()
