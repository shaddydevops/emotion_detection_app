import requests
import json
from typing import Dict, Any


WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

EMOTION_KEYS = ["anger", "disgust", "fear", "joy", "sadness"]


def _find_emotion_scores(payload: Any) -> Dict[str, float]:
    """
    Traverse a JSON-like structure and return a dict with the five standard
    emotions if found. Missing keys default to 0.0.
    This makes the parsing robust to minor structural changes.
    """

    if isinstance(payload, dict):
        if all(k in payload for k in EMOTION_KEYS):
            return {k: float(payload.get(k, 0.0)) for k in EMOTION_KEYS}

   
        for v in payload.values():
            found = _find_emotion_scores(v)
            if found:
                return found

    if isinstance(payload, list):
        for item in payload:
            found = _find_emotion_scores(item)
            if found:
                return found

    return {}  # not found here


def emotion_detector(text_to_analyze: str) -> Dict[str, float | str]:
    """
    Calls the Watson NLP EmotionPredict function, parses the JSON response,
    extracts anger/disgust/fear/joy/sadness, computes dominant_emotion,
    and returns the formatted dictionary:

    {
      'anger': ...,
      'disgust': ...,
      'fear': ...,
      'joy': ...,
      'sadness': ...,
      'dominant_emotion': 'joy'
    }
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    resp = requests.post(WATSON_EMOTION_URL, headers=HEADERS, json=payload, timeout=15)

    # 1) Convert response text → dict
    try:
        data = json.loads(resp.text)  # per instructions, parse from response.text
    except json.JSONDecodeError:
        # If the service doesn’t return valid JSON, return a safe default
        return {
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "dominant_emotion": "unknown",
        }

    # 2) Find the five scores wherever they are nested
    scores = _find_emotion_scores(data)

    # 3) Fill defaults for any missing keys
    result = {k: float(scores.get(k, 0.0)) for k in EMOTION_KEYS}

    # 4) Compute dominant_emotion
    dominant = max(result, key=result.get) if any(result.values()) else "unknown"
    result["dominant_emotion"] = dominant

    return result


# Optional: quick manual test when running this file directly
if __name__ == "__main__":
    sample = "I am so happy I am doing this."
    print(emotion_detector(sample))
