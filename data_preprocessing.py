```python
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(raw_data):
    """
    Function to preprocess the raw data.
    :param raw_data: DataFrame, the raw data
    :return: DataFrame, the preprocessed data
    """
    # Drop any rows with missing values
    data = raw_data.dropna()

    # Convert categorical variables into dummy/indicator variables (i.e., one-hot encoding)
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols)

    # Normalize numerical variables
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data

if __name__ == "__main__":
    # Load the raw data
    raw_data = pd.read_csv('raw_data.csv')

    # Preprocess the data
    preprocessed_data = preprocess_data(raw_data)

    # Save the preprocessed data
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)
```
