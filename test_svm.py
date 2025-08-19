from svm import classifier

def test_prediction():
    # Check that the classifier predicts either 0 or 1 for input 55
    result = classifier.predict([[55]])
    assert result[0] in [0, 1]
