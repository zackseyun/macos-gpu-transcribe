import unittest

from format_text import format_transcription


class BrandReplacementTest(unittest.TestCase):
    def test_corrects_standalone_kartha_to_cartha(self):
        self.assertEqual(
            format_transcription("kartha keeps getting transcribed with a k."),
            "Cartha keeps getting transcribed with a k.",
        )

    def test_corrects_kartha_domain_forms_before_standalone_brand(self):
        self.assertEqual(
            format_transcription("open kartha dot ai dot mobile and kartha website"),
            "Open cartha.ai.mobile and cartha.website",
        )

    def test_corrects_kartha_service_name(self):
        self.assertEqual(
            format_transcription("check kartha cdk service logs"),
            "Check CarthaCdkService logs",
        )

    def test_does_not_touch_other_karth_names(self):
        self.assertEqual(
            format_transcription("Karthik can review it later."),
            "Karthik can review it later.",
        )

    def test_corrects_zach_to_zack(self):
        self.assertEqual(
            format_transcription("zach should review the app."),
            "Zack should review the app.",
        )

    def test_does_not_touch_longer_zach_names(self):
        self.assertEqual(
            format_transcription("Zachary can review it later."),
            "Zachary can review it later.",
        )


if __name__ == "__main__":
    unittest.main()
