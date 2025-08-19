import unittest
from EmotionDetection import emotion_detector

class TestEmotionDetection(unittest.TestCase):
    """
    Unit tests for the EmotionDetection package.
    Each test checks that the dominant_emotion matches the expected label
    for the provided statement.
    """

    def test_emotion_joy(self):
        text = "I am glad this happened"
        result = emotion_detector(text)
        print("JOY result:", result)  # visible in terminal output screenshot
        self.assertIsInstance(result, dict)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"].lower(), "joy")

    def test_emotion_anger(self):
        text = "I am really mad about this"
        result = emotion_detector(text)
        print("ANGER result:", result)
        self.assertIsInstance(result, dict)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"].lower(), "anger")

    def test_emotion_disgust(self):
        text = "I feel disgusted just hearing about this"
        result = emotion_detector(text)
        print("DISGUST result:", result)
        self.assertIsInstance(result, dict)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"].lower(), "disgust")

    def test_emotion_sadness(self):
        text = "I am so sad about this"
        result = emotion_detector(text)
        print("SADNESS result:", result)
        self.assertIsInstance(result, dict)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"].lower(), "sadness")

    def test_emotion_fear(self):
        text = "I am really afraid that this will happen"
        result = emotion_detector(text)
        print("FEAR result:", result)
        self.assertIsInstance(result, dict)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"].lower(), "fear")

if __name__ == "__main__":
    unittest.main(verbosity=2)
