import pickle

# Load the trained model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Get input from the user
news = input("Enter a news article: ")

# Convert the input text into TF-IDF features
news_tfidf = vectorizer.transform([news])

# Predict
prediction = model.predict(news_tfidf)

# Get confidence scores
probability = model.predict_proba(news_tfidf)

# Display the result
if prediction[0] == 0:
    print("\nPrediction: Fake News")
    print(f"Confidence: {probability[0][0] * 100:.2f}%")
else:
    print("\nPrediction: Real News")
    print(f"Confidence: {probability[0][1] * 100:.2f}%")