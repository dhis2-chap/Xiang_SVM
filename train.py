import argparse

import joblib
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os

def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location] for location in unique_locations_list}
    return locations

def fill_disease_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in disease case data by:
    1. Forward filling,
    2. Backward filling,
    3. Filling remaining NaNs with 0.
    """
    return df.ffill().bfill().fillna(0)

def train(csv_fn, model_fn):
    models = {}
    locations = get_df_per_location(csv_fn)
    for location, data in locations.items():

        data = pd.read_csv(csv_fn)
        #data['data_iniSE'] = pd.to_datetime(data['data_iniSE'])
        data['time_period'] = pd.to_datetime(data['time_period'])
        data = data[['time_period', 'disease_cases']]
        data = fill_disease_data(data)

        #assert False, os.path.abspath(csv_fn)


        # Initialize the scaler and SVR model
        scaler = StandardScaler()
        svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

        # Training data for the current window
        number_data_points = data.shape[0]

        X_train = data['disease_cases'][:number_data_points - 1].values.reshape(-1, 1)
        y_train = data['disease_cases'][1:number_data_points].values  # Shift target by 1 to predict the next step

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit the model
        svm_model.fit(X_train_scaled, y_train)

        models[location] = svm_model

    joblib.dump(models, model_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)


