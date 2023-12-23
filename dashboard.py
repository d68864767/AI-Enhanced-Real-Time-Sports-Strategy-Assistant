```python
# Import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from keras.models import load_model
from strategy_engine import predict_outcomes, recommend_strategy

# Load the trained model
model = load_model('model.h5')

# Load the preprocessed data
preprocessed_data = pd.read_csv('preprocessed_data.csv')

# Split the data into features
X = preprocessed_data.drop('target', axis=1)

# Predict the outcomes
predictions = predict_outcomes(model, X)

# Recommend a strategy
strategy = recommend_strategy(predictions)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1('AI-Enhanced Real-Time Sports Strategy Assistant'),
    html.Div([
        html.H2('Predicted Outcomes'),
        dcc.Graph(
            id='predicted-outcomes',
            figure={
                'data': [
                    {'x': list(range(len(predictions))), 'y': predictions.flatten(), 'type': 'scatter', 'name': 'Outcome'}
                ],
                'layout': {
                    'title': 'Predicted Outcomes Over Time'
                }
            }
        )
    ]),
    html.Div([
        html.H2('Recommended Strategy'),
        html.P(id='recommended-strategy', children=f'Recommended Strategy: {strategy}')
    ])
])

# Define the app callbacks
@app.callback(
    Output('recommended-strategy', 'children'),
    [Input('predicted-outcomes', 'clickData')]
)
def update_recommended_strategy(clickData):
    # Update the recommended strategy based on the latest predicted outcome
    if clickData is not None:
        latest_prediction = clickData['points'][0]['y']
        strategy = 'Offensive' if latest_prediction > 0.5 else 'Defensive'
        return f'Recommended Strategy: {strategy}'

    return dash.no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```
