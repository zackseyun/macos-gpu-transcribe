import unittest

import transcribe


class ModelDefaultsTest(unittest.TestCase):
    def test_fn_default_model_is_cohere_swift_4bit(self):
        self.assertEqual(transcribe.DEFAULT_MODEL_MODE, "cohere-swift-4bit")
        self.assertEqual(transcribe.MENU_MODEL_MODES[0], "cohere-swift-4bit")
        self.assertEqual(transcribe.MODEL_LABELS["fast"], "Qwen3-ASR 0.6B 4-bit")
        self.assertIn("cohere", transcribe.MENU_MODEL_MODES)
        self.assertIn("fast", transcribe.MENU_MODEL_MODES)
        self.assertIn("cohere-swift-4bit", transcribe.MENU_MODEL_MODES)
        self.assertIn("cohere-pytorch", transcribe.MENU_MODEL_MODES)
        self.assertEqual(transcribe.MODEL_LABELS["cohere"], "Cohere Transcribe MLX 8-bit")
        self.assertEqual(
            transcribe.MODEL_LABELS["cohere-swift-4bit"],
            "Cohere Transcribe Swift 4-bit",
        )


if __name__ == "__main__":
    unittest.main()
