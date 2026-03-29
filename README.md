# IMDb Sentiment Analyzer (BiLSTM)

This project is a  sentiment analysis web app built using a Bidirectional LSTM model. It takes a movie review as input and predicts whether the sentiment is positive or negative.

I built this mainly to understand how LSTM-based models work on text data and how to deploy them using Streamlit.

---
# Live Demo

You can try the app here:
https://imdb-sentiment-analysis-1.streamlit.app/

---

## Project Structure

```

├── models/
│   ├── model.pth
│   └── vocab.pkl
├── .gitignore
├── Requirements.txt
├── app.py
├── config.py
├── imdb_sentiment_analysis.ipynb
├── sample_review.py


```

---

## What the App Does

* Takes a movie review as input
* Cleans and preprocesses the text
* Converts words into numerical format using a saved vocabulary
* Runs the input through a trained BiLSTM model
* Outputs:

  * Sentiment (Positive / Negative)
  * Confidence score

---

## Model Details

* Embedding Layer
* 2-layer Bidirectional LSTM
* Dropout for regularization
* Fully connected layer with sigmoid activation

Key configuration:

* Max sequence length: 300
* Embedding dimension: 128
* Hidden dimension: 128
* Threshold: 0.77

---

## How to Run

1. Clone the repository

```
git clone <your-repo-link>
cd streamlit-app
```

2. Install dependencies

```
pip install -r Requirements.txt
```

3. Run the app

```
streamlit run app.py
```

---

## Features

* Simple and clean UI using Streamlit
* Example reviews for quick testing
* Real-time prediction
* Confidence score displayed

---

## Notebook

The file `imdb_sentiment_analysis.ipynb` contains the full training process including:

* Data preprocessing
* Model building
* Training and evaluation

---

## Limitations

* Model is trained only on IMDb-style reviews
* Struggles with sarcasm in some cases
* Limited vocabulary (uses OOV token for unknown words)

---

## Future Improvements

* Try pretrained embeddings (GloVe / Word2Vec)
* Replace BiLSTM with BERT or other transformer models
* Add neutral sentiment class
* Deploy the app online

---

## Author

Manish Mahara
