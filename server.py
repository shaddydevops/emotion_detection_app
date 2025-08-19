"""Flask server for Watson NLP emotion detection.

Serves the provided UI (index.html + mywebscript.js) and exposes the
/emotionDetector endpoint (as required by the project) which accepts text input
and returns a formatted analysis string.
"""

from __future__ import annotations

from typing import Dict, Tuple
from flask import Flask, request, render_template
from EmotionDetection import emotion_detector


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Flask: Configured Flask app pointing to the provided templates/static.
    """
    app = Flask(
        __name__,
        template_folder="oaqjp-final-project-emb-ai/templates",
        static_folder="oaqjp-final-project-emb-ai/static",
    )

    @app.route("/")
    def index() -> str:
        """Render the landing page (provided in the repo)."""
        return render_template("index.html")

    # Keep the required URL path /emotionDetector, but use a snake_case function
    # name to satisfy pylint. We set endpoint="emotionDetector" to expose the
    # endpoint name exactly as specified by the assignment.
    @app.route("/emotionDetector", methods=["GET", "POST"], endpoint="emotionDetector")
    def emotion_detector_route() -> Tuple[str, int, Dict[str, str]]:
        """Analyze text and return a formatted result string.

        Accepts:
            - GET query param: textToAnalyze
            - POST form field: textToAnalyze
            - POST JSON: {"text": "..."} (fallback)

        Returns:
            tuple[str, int, dict[str, str]]: (body, status_code, headers)
        """
        # Prefer GET query param if present (matches provided frontend behavior)
        text = request.args.get("textToAnalyze")

        # Fallbacks for POST form/JSON
        if not text:
            text = request.form.get("textToAnalyze")
        if not text and request.is_json and request.json:
            text = request.json.get("text")

        # Call detector even if text is blank; detector handles blank/error cases.
        scores = emotion_detector(text or "")

        # If dominant_emotion is None, show the required error message.
        if scores.get("dominant_emotion") is None:
            return (
                "Invalid text! Please try again!",
                200,
                {"Content-Type": "text/plain; charset=utf-8"},
            )

        anger = scores.get("anger", 0.0)
        disgust = scores.get("disgust", 0.0)
        fear = scores.get("fear", 0.0)
        joy = scores.get("joy", 0.0)
        sadness = scores.get("sadness", 0.0)
        dominant = scores.get("dominant_emotion", "unknown")

        response_text = (
            "For the given statement, the system response is "
            f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, "
            f"'joy': {joy} and 'sadness': {sadness}. "
            f"The dominant emotion is {dominant}."
        )
        return response_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    return app


app = create_app()

if __name__ == "__main__":
    # Run on localhost:5000 as required
    app.run(host="0.0.0.0", port=5000, debug=True)
