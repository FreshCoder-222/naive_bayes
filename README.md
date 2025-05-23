# Navie Bayes

## Spam Detection Project
This project shows how to use the Naive Bayes algorithm to detect spam messages.

## Files

- `naive_bayes_codes.py` - Uses a real dataset to train and test a spam detection model.
- `naive_bayes_code_1.py` - A small example to understand how the model works.

## How to Use

1. Make sure you have Python installed.
2. Install the required libraries: 
``` bash
pip install pandas scikit-learn
```
3. Put the `spam.csv` file in the same folder as the code.
4. Run the code using: 

```bash
naive_bayes_codes.py 
naive_bayes_code_1.py
```

This project is for learning and practice.

-----
-----
# Small Dataset Programs


# Naive Bayes Classifier Projects

This repository contains multiple projects demonstrating the use of different types of Naive Bayes classifiers using `scikit-learn`:

- **Gaussian Naive Bayes (GNB)**
- **Multinomial Naive Bayes (MNB)**
- **Bernoulli Naive Bayes (BNB)**

Each program is designed for simple classification tasks using different types of data such as numeric, word counts, and binary word presence.

---

##  Project Structure
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

# Run each script using Python:

```bash
python gaussian_navie_bayes.py
python multinomial_navie_bayes.py
python bernoulli_navie_bayes.py
```


## What You'll Learn

* Difference between GaussianNB, MultinomialNB, and BernoulliNB

* How to handle different types of input features:

1. Continuous values (Gaussian)

2. Count-based and discrete values (Multinomial)

3. Binary presence ( 0 or 1 , present or absent and ture or false) (Bernoulli)

* Accuracy and prediction interpretation

* How to build and train a model
----

# Large Dataset Programs

This section contains programs that use large, real-world datasets to apply and test Naive Bayes classifiers. These examples help understand how Naive Bayes works on practical data.

## Programs Included

### 1. Breast Cancer Detection
- **Model Used**: Gaussian Naive Bayes
- **Dataset**: Breast Cancer Wisconsin
- **Goal**: Predict whether a tumor is benign or malignant
- **Feature Type**: Numerical values (e.g., size, texture)
- **Output**: Model accuracy (e.g., Accuracy: 0.94)

### 2. News Article Classification
- **Model Used**: Multinomial Naive Bayes
- **Dataset**: 20 Newsgroups
- **Goal**: Classify articles into different topics
- **Feature Type**: Word frequencies
- **Output**: Model accuracy (e.g., Accuracy: 0.98)

### 3. SMS Spam Detection
- **Model Used**: Bernoulli Naive Bayes
- **Dataset**: SMS Spam Collection
- **Goal**: Identify whether a message is spam or not
- **Feature Type**: Binary word presence (0 or 1)
- **Output**: Model accuracy (e.g., Accuracy: 0.98)
- **Note**: Confirm that you placed the dataset (SMSSpamCollection.csv)

---

## How to Run the Programs

1. Install the required libraries (if not already installed):
   ```bash
   pip install scikit-learn pandas
   ```

------

## What You'll Learn

* How to use Naive Bayes on large datasets

* When to use:

1.     GaussianNB → for numeric data

2.     MultinomialNB → for count-based text data

3.     BernoulliNB → for binary text data

* How to interpret model accuracy and predictions





