from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


# Sample data
texts = ["Breaking news: the economy is crashing", "This is fake news", "Scientists discover cure", "You won a prize", "Government confirms report"]
labels = [0, 1, 0, 1, 0]  # 1 = fake, 0 = real

# Vectorize using binary features
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and evaluate
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("Fake news Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))



# for seperating the program 
print("\n" + "-"*50 + "\n")




from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Features: [buy, free, win, click, hello, congratulations]
X = [
    [1, 1, 1, 0, 0, 1],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
    [1, 1, 0, 1, 0, 0],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
    [0, 0, 0, 1, 0, 1],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
]

y = [1, 0, 1, 0, 1, 0]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print ("Small Spam Detection Dataset Output")
print("\n Predicted:", y_pred)
print("\n Actual:   ", y_test)
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))



# for seperating the program 
print("\n" + "-"*50 + "\n")




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load SMS dataset
df = pd.read_csv("/home/stemland/sms_data/spam_detection/SMSSpamCollection.csv", sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print ("UCI SMS Spam Collection Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

