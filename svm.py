import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

# Training data
X = [[30], [40], [50], [60], [20], [10], [70]]
y = [0, 1, 1, 1, 0, 0, 1]

# Train the model
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X, y)

def main():
    """Run a sample prediction"""
    X_marks = [[55]]
    prediction = classifier.predict(X_marks)
    print("Prediction for 55:", prediction[0])

if __name__ == "__main__":
    main()
