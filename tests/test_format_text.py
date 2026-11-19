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

    def test_corrects_michelle_to_mishaal(self):
        self.assertEqual(
            format_transcription("michelle should review it later."),
            "Mishaal should review it later.",
        )

    def test_corrects_quinn_to_qwen(self):
        self.assertEqual(
            format_transcription("the quinn branch feels faster."),
            "The Qwen branch feels faster.",
        )
        self.assertEqual(
            format_transcription("the quin branch feels faster."),
            "The Qwen branch feels faster.",
        )
        self.assertEqual(
            format_transcription("the quen branch feels faster."),
            "The Qwen branch feels faster.",
        )
        self.assertEqual(
            format_transcription("this QIN model seems strong."),
            "This Qwen model seems strong.",
        )
        self.assertEqual(
            format_transcription("this Quan MLX model seems strong."),
            "This Qwen MLX model seems strong.",
        )
        self.assertEqual(
            format_transcription("switch to quinn three asr."),
            "Switch to Qwen3-ASR.",
        )
        self.assertEqual(
            format_transcription("switch to QIN three ASR."),
            "Switch to Qwen3-ASR.",
        )

    def test_corrects_video_model_names(self):
        self.assertEqual(
            format_transcription("compare c dance and cling for this video."),
            "Compare Seedance and Kling for this video.",
        )
        self.assertEqual(
            format_transcription("try sea dance after seeddance."),
            "Try Seedance after Seedance.",
        )
        self.assertEqual(
            format_transcription("try a different model like clang."),
            "Try a different model like Kling.",
        )
        self.assertEqual(
            format_transcription("seeddance and kling should be spelled correctly."),
            "Seedance and Kling should be spelled correctly.",
        )


class SpacingRepairTest(unittest.TestCase):
    def test_repairs_missing_space_before_iphone(self):
        self.assertEqual(
            format_transcription("can you run Flutter run on myiPhone?"),
            "Can you run Flutter run on my iPhone?",
        )

    def test_repairs_inserted_space_inside_iphone(self):
        self.assertEqual(
            format_transcription("can you run flutter run on my i phone?"),
            "Can you run flutter run on my iPhone?",
        )

    def test_repairs_common_spaced_developer_terms(self):
        self.assertEqual(
            format_transcription("the swift ui x code code base was pre launched."),
            "The SwiftUI Xcode codebase was pre-launch.",
        )

    def test_repairs_common_spaced_video_words(self):
        self.assertEqual(
            format_transcription("update the out ros because this got ner fed."),
            "Update the outros because this got nerfed.",
        )

    def test_normalizes_spaced_ampm_only_after_time(self):
        self.assertEqual(
            format_transcription("make sure friday at 6 p. m. eastern is bigger."),
            "Make sure friday at 6 p.m. eastern is bigger.",
        )
        self.assertEqual(
            format_transcription("i am still working."),
            "I am still working.",
        )


class NumberConversionTest(unittest.TestCase):
    def assertFormats(self, spoken, expected):
        self.assertEqual(format_transcription(spoken), expected)

    def test_twenty_four_seven_is_the_idiom_not_87(self):
        self.assertFormats("Yes, twenty-four seven.", "Yes, 24/7.")
        self.assertFormats("we need people on twenty four seven", "We need people on 24/7")

    def test_runs_that_are_not_one_number_keep_their_words(self):
        # word2number did arithmetic on these: 1,524 / 10 / 5 / "20 days".
        self.assertFormats("nineteen eighty four", "Nineteen eighty four")
        self.assertFormats("call nine one one", "Call nine one one")
        self.assertFormats("one two three", "One two three")
        self.assertFormats(
            "open twenty-four seven, three sixty five days a year.",
            "Open 24/7, three sixty five days a year.",
        )
        self.assertFormats("a hundred and twenty", "A hundred and twenty")

    def test_real_numbers_still_become_digits(self):
        self.assertFormats("twenty five", "25")
        self.assertFormats("twenty-one days", "21 days")
        self.assertFormats("two hundred five", "205")
        self.assertFormats("one thousand two hundred thirty four", "1,234")
        self.assertFormats("fifteen hundred", "1,500")
        self.assertFormats("twenty five percent", "25%")
        self.assertFormats("It costs five hundred dollars per month.", "It costs $500 per month.")
        self.assertFormats("About two hundred and fifty K monthly active users.", "About 250K monthly active users.")
        self.assertFormats("three x faster", "3x faster")
        self.assertFormats("Twenty-five, thirty, forty.", "25, 30, 40.")

    def test_lone_one_stays_a_word(self):
        self.assertFormats("No one knows.", "No one knows.")
        self.assertFormats("this one will just continue", "This one will just continue")
        self.assertFormats("throw one of these creator nights", "Throw one of these creator nights")
        self.assertFormats("For this one, there's one screen", "For this one, there's one screen")
        self.assertFormats("one to one meeting", "One to one meeting")

    def test_one_as_a_quantity_becomes_a_digit(self):
        self.assertFormats("one percent of users", "1% of users")
        self.assertFormats("it costs one dollar", "It costs $1")
        self.assertFormats("meet at one p.m.", "Meet at 1 p.m.")
        self.assertFormats("one or two things", "1 or 2 things")
        self.assertFormats("one, two, three", "1, 2, 3")
        self.assertFormats("one hundred people", "100 people")


if __name__ == "__main__":
    unittest.main()
