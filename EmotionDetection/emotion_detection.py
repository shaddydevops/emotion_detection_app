import requests
import json
from typing import Dict, Any

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

EMOTION_KEYS = ["anger", "disgust", "fear", "joy", "sadness"]

# Helper: a None-filled result per task spec for invalid/blank input
def _none_result() -> Dict[str, Any]:
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

def _find_emotion_scores(payload: Any) -> Dict[str, float]:
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
    return {}

def emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    # Always send a string (blank allowed)
    payload = {"raw_document": {"text": (text_to_analyze or "")}}

    try:
        resp = requests.post(WATSON_EMOTION_URL, headers=HEADERS, json=payload, timeout=15)
    except Exception:
        # Network/transport error → treat as invalid for this task
        return _none_result()

    # Task 7 requirement: if the service returns 400, return None-values
    if resp.status_code == 400:
        return _none_result()

    # Parse JSON body
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        return _none_result()

    # Extract emotion scores
    scores = _find_emotion_scores(data)
    if not scores:
        return _none_result()

    result = {k: float(scores.get(k, 0.0)) for k in EMOTION_KEYS}
    dominant = max(result, key=result.get) if any(result.values()) else None
    result["dominant_emotion"] = dominant
    return result

if __name__ == "__main__":
    print(emotion_detector(""))
