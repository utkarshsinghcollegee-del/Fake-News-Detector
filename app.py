from flask import Flask, render_template, request
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Store last 5 predictions
history = []

# Load model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # Convert to TF-IDF
    news_tfidf = vectorizer.transform([news])

    # Predict
    prediction = model.predict(news_tfidf)
    probability = model.predict_proba(news_tfidf)

    if prediction[0] == 0:
        result = "Fake News"
        confidence = probability[0][0] * 100
    else:
        result = "Real News"
        confidence = probability[0][1] * 100

    # Save history
    history.insert(0, {
        "prediction": result,
        "confidence": f"{confidence:.2f}",
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })

    # Keep only last 5
    history[:] = history[:5]

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence:.2f}",
        history=history
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)