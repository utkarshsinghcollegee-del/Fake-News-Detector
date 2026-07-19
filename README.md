# 📰 Fake News Detector

A Machine Learning web application that detects whether a news article is **Fake** or **Real**.

## Features

- Detects Fake and Real news
- Uses TF-IDF Vectorization
- Logistic Regression Classifier
- Flask Web Application
- Displays prediction confidence

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML/CSS

## Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

Download it here:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, extract the ZIP file and place the two CSV files in the `data` folder:

```text
data/
├── Fake.csv
└── True.csv
```

## Project Structure

```
Fake-News-Detector/
│── data/
│── model/
│── templates/
│── app.py
│── train.py
│── predict.py
│── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

## Model

- TF-IDF Vectorizer
- Logistic Regression

## Author

Utkarsh Singh