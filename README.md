# navie_bayes

# Spam Detection Project

This project shows how to use the Naive Bayes algorithm to detect spam messages.

## Files

- `naive_bayes_codes.py` - Uses a real dataset to train and test a spam detection model.
- `naive_bayes_code_1.py` - A small example to understand how the model works.

## How to Use

1. Make sure you have Python installed.
2. Install the required libraries: pip install pandas scikit-learn
3. Put the `spam.csv` file in the same folder as the code.
4. Run the code using: naive_bayes_codes.py and naive_bayes_code_1.py

This project is for learning and practice.
<<<<<<< HEAD



# this is for gaussian_navie_bayes, multinomial_navie_bayes and bernoulli_navie_bayes


# Naive Bayes Classifier Projects

This repository contains multiple projects demonstrating the use of different types of Naive Bayes classifiers using `scikit-learn`:

- **Gaussian Naive Bayes (GNB)**
- **Multinomial Naive Bayes (MNB)**
- **Bernoulli Naive Bayes (BNB)**

Each program is designed for simple classification tasks using different types of data such as numeric, word counts, and binary word presence.

---

##  Project Structure


---

## Descriptions

### 1. `gaussian_navie_bayes.py`

- **Dataset**: Iris flower dataset and a small weather dataset
- **Model**: `GaussianNB`
- **Purpose**: Classify flowers or weather decisions based on continuous features like petal length, temperature, etc.

### 2. `multinomial_navie_bayes.py`

- **Dataset**: Small set of movie reviews
- **Model**: `MultinomialNB`
- **Purpose**: Sentiment analysis on reviews using word frequencies

### 3. `bernoulli_navie_bayes.py`

- **Dataset**: Fake news detection and spam filtering with binary features
- **Model**: `BernoulliNB`
- **Purpose**: Classify binary text features (yes/no word presence)

---

## How to Run

* Install required libraries:
   ```bash
   pip install scikit-learn pandas

## Run each script using Python:

* python gaussian_navie_bayes.py
* python multinomial_navie_bayes.py
* python bernoulli_navie_bayes.py

## What You'll Learn

* Difference between GaussianNB, MultinomialNB, and BernoulliNB

* How to handle different types of input features:

1. Continuous values (Gaussian)

2. Count-based and discrete values (Multinomial)

3. Binary presence ( 0 or 1 , present or absent and ture or false) (Bernoulli)

* Accuracy and prediction interpretation

* How to build and train a model
