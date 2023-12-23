```python
# Import necessary libraries
import pandas as pd
from keras.models import load_model

def predict_outcomes(model, X):
    """
    Function to predict the outcomes using the deep learning model.
    :param model: Keras model, the trained model
    :param X: DataFrame, the data to predict on
    :return: array, the predicted outcomes
    """
    # Predict the outcomes
    predictions = model.predict(X)

    return predictions

def recommend_strategy(predictions):
    """
    Function to recommend a strategy based on the predicted outcomes.
    :param predictions: array, the predicted outcomes
    :return: str, the recommended strategy
    """
    # Define the strategy based on the predictions
    if predictions.mean() > 0.5:
        strategy = 'Offensive'
    else:
        strategy = 'Defensive'

    return strategy

if __name__ == "__main__":
    # Load the preprocessed data
    preprocessed_data = pd.read_csv('preprocessed_data.csv')

    # Split the data into features and labels
    X = preprocessed_data.drop('target', axis=1)

    # Load the trained model
    model = load_model('model.h5')

    # Predict the outcomes
    predictions = predict_outcomes(model, X)

    # Recommend a strategy
    strategy = recommend_strategy(predictions)

    print(f'Recommended Strategy: {strategy}')
```
