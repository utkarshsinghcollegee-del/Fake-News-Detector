from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    news_tfidf = vectorizer.transform([news])

    prediction = model.predict(news_tfidf)
    probability = model.predict_proba(news_tfidf)

    if prediction[0] == 0:
        result = "Fake News"
        confidence = probability[0][0] * 100
    else:
        result = "Real News"
        confidence = probability[0][1] * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)