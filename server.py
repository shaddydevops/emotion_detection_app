from flask import Flask, request, render_template
from EmotionDetection import emotion_detector  # from your package

# Create the Flask app
app = Flask(__name__, template_folder="oaqjp-final-project-emb-ai/templates",
            static_folder="oaqjp-final-project-emb-ai/static")

@app.route("/")
def index():
    # Serve the provided index.html
    return render_template("index.html")

@app.route("/emotionDetector", methods=["GET", "POST"])
def emotionDetector():
    # Prefer GET query param if present (matches mywebscript.js behavior)
    text = request.args.get("textToAnalyze")

    # Fallbacks for POST form/JSON
    if not text:
        text = request.form.get("textToAnalyze")
    if not text and request.is_json and request.json:
        text = request.json.get("text")

    if not text or not text.strip():
        return "Please provide text to analyze.", 400

    scores = emotion_detector(text)

    anger   = scores.get("anger", 0.0)
    disgust = scores.get("disgust", 0.0)
    fear    = scores.get("fear", 0.0)
    joy     = scores.get("joy", 0.0)
    sadness = scores.get("sadness", 0.0)
    dominant = scores.get("dominant_emotion", "unknown")

    response_text = (
        f"For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, 'joy': {joy} "
        f"and 'sadness': {sadness}. The dominant emotion is {dominant}."
    )
    return response_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

if __name__ == "__main__":
    # Run on localhost:5000 as required
    app.run(host="0.0.0.0", port=5000, debug=True)
