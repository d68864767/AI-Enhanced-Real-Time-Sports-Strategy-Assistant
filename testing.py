```python
# Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model
from data_preprocessing import preprocess_data

def load_test_data():
    """
    Function to load the test data.
    :return: DataFrame, the loaded data
    """
    # Load the test data
    # This is a placeholder and should be replaced with the actual function to load test data
    test_data = pd.read_csv('test_data.csv')

    return test_data

def evaluate_model(model, X_test, y_test):
    """
    Function to evaluate the performance of the deep learning model.
    :param model: Keras model, the trained model
    :param X_test: DataFrame, the test data
    :param y_test: Series, the test labels
    :return: float, the accuracy of the model
    """
    # Predict the outcomes
    y_pred = model.predict_classes(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix
    print(confusion_matrix(y_test, y_pred))

    return accuracy

if __name__ == "__main__":
    # Load the test data
    test_data = load_test_data()

    # Preprocess the test data
    preprocessed_test_data = preprocess_data(test_data)

    # Split the test data into features and labels
    X_test = preprocessed_test_data.drop('target', axis=1)
    y_test = preprocessed_test_data['target']

    # Load the trained model
    model = load_model('model.h5')

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Print the accuracy
    print(f"Model Accuracy: {accuracy}")
```
