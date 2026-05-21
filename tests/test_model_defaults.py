import unittest

import transcribe


class ModelDefaultsTest(unittest.TestCase):
    def test_fn_default_model_is_qwen_fast(self):
        self.assertEqual(transcribe.DEFAULT_MODEL_MODE, "fast")
        self.assertEqual(transcribe.MENU_MODEL_MODES[0], "fast")
        self.assertEqual(transcribe.MODEL_LABELS["fast"], "Qwen3-ASR 0.6B 4-bit")
        self.assertIn("cohere", transcribe.MENU_MODEL_MODES)
        self.assertIn("cohere-pytorch", transcribe.MENU_MODEL_MODES)
        self.assertEqual(transcribe.MODEL_LABELS["cohere"], "Cohere Transcribe MLX 8-bit")


if __name__ == "__main__":
    unittest.main()
