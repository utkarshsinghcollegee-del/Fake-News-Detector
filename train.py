import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
# Load the datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
print("Fake News Dataset:")
print(fake.head())

print("\nTrue News Dataset:")
print(true.head())
# Add labels
fake["label"] = 0
true["label"] = 1
print(fake[["title", "label"]].head())
print(true[["title", "label"]].head())
# Combine both datasets
data = pd.concat([fake, true], axis=0)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42)

# Reset the index
data.reset_index(drop=True, inplace=True)
print("\nCombined Dataset:")
print(data.head())

print("\nTotal rows:", len(data))
# Select input and output
X = data["text"]
y = data["label"]
print("\nFirst News Article:\n")
print(X.iloc[0])

print("\nLabel:", y.iloc[0])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# Create the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Convert text into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\nTF-IDF Training Shape:", X_train_tfidf.shape)
print("TF-IDF Testing Shape:", X_test_tfidf.shape)
# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
# Save the trained model
with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the TF-IDF vectorizer
with open("model/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")