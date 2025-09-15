import argparse

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location] for location in unique_locations_list}
    return locations

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    number_of_weeks_pred = 3
    models = joblib.load(model_fn)
    locations_future = get_df_per_location(future_climatedata_fn)
    #print("HERE: ", future_climatedata_fn)
    #assert False, os.path.abspath(future_climatedata_fn)
    locations_historic = get_df_per_location(historic_data_fn)
    first_location = True

    for location, df in locations_future.items():

        data = locations_historic[location]

        data = data.fillna(0)

        number_data_points = data.shape[0]

        model = models[location]

        X = data.iloc[number_data_points - number_of_weeks_pred - 1]['disease_cases'].reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit_transform(data['disease_cases'][:number_data_points - (number_of_weeks_pred + 1)].values.reshape(-1, 1))


        predictions = []

        for i in range(number_of_weeks_pred):
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)[0]
            predictions.append(y_pred)
            X = y_pred.reshape(-1, 1)

        df['sample_0'] = np.array(predictions)

        if first_location:
            df.to_csv(predictions_fn, index=False, mode='w', header=True)
            first_location = False
        else:
            df.to_csv(predictions_fn, index=False, mode='a', header=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
