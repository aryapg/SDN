import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

# Training data
X = [[30], [40], [50], [60], [20], [10], [70]]
y = [0, 1, 1, 1, 0, 0, 1]

# Train classifier
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X, y)

def predict_value(val):
    """Return prediction for a single input value"""
    return classifier.predict([[val]])[0]

def main():
    X_marks = [[55]]
    print("Prediction for 55:", classifier.predict(X_marks)[0])

# ---------------------------
# Tests (pytest will discover)
# ---------------------------
def test_prediction_for_55():
    result = predict_value(55)
    assert result in [0, 1]   # should always predict valid class

def test_prediction_for_low_value():
    result = predict_value(10)
    assert result == 0        # since label for 10 was 0

def test_prediction_for_high_value():
    result = predict_value(70)
    assert result == 1        # since label for 70 was 1

if __name__ == "__main__":
    main()
