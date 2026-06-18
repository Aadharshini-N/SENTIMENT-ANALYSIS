import pickle
import re
import nltk
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from database.db import init_db, save_review, get_history

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    negation_words = {
        "not", "no", "never", "hardly", "barely",
        "nothing", "nowhere", "neither", "nor", "cannot"
    }

    custom_stopwords = stop_words - negation_words

    words = text.split()
    words = [w for w in words if w not in custom_stopwords]

    return " ".join(words)

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review_text", "").strip()

    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    cleaned = clean_text(review_text)

    vectorized = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(vectorized)[0]

    negative_prob = probabilities[0]
    positive_prob = probabilities[1]

    neutral_words = {
        "average", "okay", "ok", "fine",
        "normal", "decent", "acceptable",
        "fair", "moderate", "ordinary"
    }

    mixed_words = {
        "but", "however", "although",
        "though", "yet", "while"
    }

    words = set(cleaned.split())

    # Neutral keyword detection
    if words.intersection(neutral_words):
        sentiment = "Neutral"

    # Mixed review detection
    elif words.intersection(mixed_words):
        sentiment = "Neutral"

    # Similar probabilities = Neutral
    elif abs(positive_prob - negative_prob) < 0.30:
        sentiment = "Neutral"

    elif positive_prob > negative_prob:
        sentiment = "Positive"

    else:
        sentiment = "Negative"

    confidence = round(max(probabilities) * 100, 1)

    print("=" * 50)
    print("INPUT:", review_text)
    print("CLEANED:", cleaned)
    print("POSITIVE:", round(positive_prob, 4))
    print("NEGATIVE:", round(negative_prob, 4))
    print("SENTIMENT:", sentiment)
    print("=" * 50)

    save_review(review_text, sentiment, confidence)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "review_text": review_text
    })

@app.route("/history", methods=["GET"])
def history():
    records = get_history(limit=10)
    return jsonify(records)

if __name__ == "__main__":
    app.run(debug=True)