# program for  iris dataset 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print ("Iris Flower Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))




# for seperating the program 
print("\n" + "-"*50 + "\n")



import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Weather dataset
data = {
    'Temperature': [30, 25, 27, 20, 32, 24, 33],
    'Humidity': [85, 80, 90, 70, 95, 65, 75],
    'Play': [0, 1, 0, 1, 0, 1, 1]  # 1: Yes, 0: No
}
df = pd.DataFrame(data)

X = df[['Temperature', 'Humidity']]
y = df['Play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print ("Weather Prediction Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))







# for seperating the program 
print("\n" + "-"*50 + "\n")




from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print ("Brest Cancer Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


