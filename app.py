from flask import Flask, render_template, request

# Database Connection
import mysql.connector
from connection import connect_to_database, fetch_data_from_database

# Machine Learning Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from keras.preprocessing.sequence import TimeseriesGenerator

app = Flask(__name__)

def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def make_predictions_and_update_df(model, current_batch, n_input, n_features, num_months, new_df, scaler, start_date, scaled_extended_train, selected_province):
    predictions = []

    current_batch = scaled_extended_train[-n_input:].reshape((1, n_input, n_features))

    for i in range(num_months):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(predictions)

    # Add predictions to the DataFrame
    pred_dates = pd.date_range(start=start_date, periods=num_months, freq='MS')
    pred_df = pd.DataFrame(index=pred_dates, columns=['Predictions'])
    pred_df['Predictions'] = true_predictions

    # Extract predictions for the selected province from extended_pred_df
    selected_province_predictions = pred_df[['Predictions']].rename(columns={'Predictions': selected_province})
    
    # Add 2024 predictions to the new_df DataFrame
    existing_dates = selected_province_predictions.index.intersection(new_df.index)

    # Remove the 'Predictions' column
    #selected_province_predictions = selected_province_predictions.drop(columns=['Predictions'])

    if not existing_dates.empty:
        # If some dates already exist, update only the missing dates
        new_dates = selected_province_predictions.index.difference(new_df.index)
        new_df = pd.concat([new_df, selected_province_predictions.loc[new_dates]], axis=0)
    else:
        # If no dates exist, concatenate the entire DataFrame
        new_df = pd.concat([new_df, selected_province_predictions], axis=0)

    # Drop duplicate rows based on the index (Date)
    new_df = new_df[~new_df.index.duplicated(keep='last')]

    return new_df, pred_df

def store_predictions_in_database(predictions_df, province):
    # Store predictions in the database
    conn = connect_to_database()
    cursor = conn.cursor()

    for date, prediction in zip(predictions_df.index, predictions_df['Predictions']):
        formatted_date = date.strftime('%Y-%m-%d')

        # Check if the record already exists for the given date and province
        check_query = f"SELECT * FROM predictionstb WHERE Date = '{formatted_date}' AND Province = '{province}';"
        cursor.execute(check_query)
        existing_record = cursor.fetchone()

        if existing_record:
            # Update the existing record if it already exists
            update_query = f"UPDATE predictionstb SET Predictions = {prediction} WHERE Date = '{formatted_date}' AND Province = '{province}';"
            cursor.execute(update_query)
        else:
            # Insert a new record if it doesn't exist
            insert_query = f"INSERT INTO predictionstb (Date, Province, Predictions) VALUES ('{formatted_date}', '{province}', {prediction});"
            cursor.execute(insert_query)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

def get_prediction_pivot(conn, province, preferredYear):
    # Fetch data from the predictionstb table for the selected province
    prediction_data = fetch_data_from_database(conn, f"SELECT * FROM predictionstb WHERE Province = '{province}';")

    # Convert the fetched data to a DataFrame
    prediction_df = pd.DataFrame(prediction_data, columns=['Date', 'Province', 'Predictions'])

    # Convert Date column to datetime format
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    # Filter the DataFrame to include only the data for the year X
    prediction_df = prediction_df[prediction_df['Date'].dt.year == preferredYear]

    # Convert the 'Date' column to string in the DataFrame
    prediction_df['Date'] = prediction_df['Date'].dt.strftime('%Y-%m-%d')

    # Pivot the DataFrame for the year X
    prediction_pivot = prediction_df.pivot(index='Date', columns='Province', values='Predictions')

    # Reset index to make 'Date' a regular column again
    prediction_pivot.reset_index(inplace=True)

    return prediction_pivot

def plot_and_table(predictions_df, province, year):
    # Plot predictions for the given year
    plt.figure(figsize=(14, 5))
    plt.plot(predictions_df.index, predictions_df['Predictions'], label=f'Predicted Data for {year}', linestyle='--', color='green')
    plt.title(f'Predicted Prices for {year}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    predicted_plot_path = f'static/predicted_{year.lower()}.png'
    plt.savefig(predicted_plot_path)
    plt.close()

    # Display the table for the given year
    table_data = predictions_df.reset_index().to_dict(orient='records')

    return predicted_plot_path, table_data
    
@app.route('/', methods=['GET', 'POST'])
def index():
    # Connect to MySQL and fetch data using mysql-connector-python
    # Connect to the database
    conn = connect_to_database()

    # SQL query to fetch data
    sql_query = "SELECT * FROM swinetb;"

    # Fetch data from the database
    data = fetch_data_from_database(conn, sql_query)

    # Convert the fetched data to a DataFrame
    df = pd.DataFrame(data, columns=['id','Date','PHILIPPINES','Cordillera','Abra','Apayao','Benguet','Ifugao','Kalinga','MountainProvince','REGIONI',
                                        'IlocosNorte','IlocosSur','LaUnion','Pangasinan','REGIONII','Batanes','Cagayan','Isabela','NuevaVizcaya','Quirino',
                                        'REGIONIII','Aurora','Bataan','Bulacan','NuevaEcija','Pampanga','Tarlac','Zambales','REGIONIVA','Batangas','Cavite','Laguna',
                                        'Quezon','Rizal','MIMAROPAREGION','Marinduque','OccidentalMindoro','OrientalMindoro','Palawan','Romblon','REGIONV','Albay',
                                        'CamarinesNorte','CamarinesSur','Catanduanes','Masbate','Sorsogon','REGIONVI','Aklan','Antique','Capiz','Guimaras','Iloilo',
                                        'NegrosOccidental','REGIONVII','Bohol','Cebu','NegrosOriental','Siquijor','REGIONVIII','Biliran','EasternSamar','Leyte',
                                        'NorthernSamar','Samar','SouthernLeyte','REGIONIX','ZamboangadelNorte','ZamboangadelSur','ZamboangaSibugay','ZamboangaCity',
                                        'REGIONX','Bukidnon','Camiguin','LanaodelNorte','MisamisOccidental','MisamisOriental','REGIONXI','DavaodeOro','DavaodelNorte',
                                        'DavaodelSur','DavaoOccidental','DavaoOriental','CityofDavao','REGIONXII','Cotabato','Sarangani','SouthCotabato','SultanKudarat',
                                        'REGIONXIII','AgusandelNorte','AgusandelSur','SurigaodelNorte','SurigaodelSur','MUSLIMMINDANAO','Basilan','Maguindanao','Sulu'])

    # Get column names for dropdown options
    columns = df.columns[2:]  # Exclude 'id' and 'Date'

    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.drop(columns=['id'], inplace=True)  # Drop the 'id' column
    df.sort_index(inplace=True)

    # Convert non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with the mean
    df.fillna(df.mean(), inplace=True)

    # Initialize ph_df with a default value
    selected_province = 'PHILIPPINES'
    ph_df = pd.DataFrame(df, columns=[selected_province])

    if request.method == 'POST':
        selected_province = request.form['province']
        # Update the data frame based on the selected province
        ph_df = pd.DataFrame(df, columns=[selected_province])

    # model filename based on selection
    model_filename = f'model/lstm_model_{selected_province.lower()}.h5'

    train = ph_df.iloc[:120]
    test = ph_df.iloc[120:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    val_data = scaled_train[96:]

    #Time Series Generator Modifications
    n_input = 18
    n_features = 1
    generator_train = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    generator_val = TimeseriesGenerator(val_data, val_data, length=n_input, batch_size=1)

    # Model Architechture
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer='l2'))

    optimizer = Adam(learning_rate=0.0001, decay=1e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

    try:
        model = load_model(model_filename)
    except:
        model.fit(generator_train, epochs=100, validation_data=generator_val, callbacks=[early_stop])
        # Save the model with the selected province in the filename
        model.save(model_filename)

    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)

    test['Predictions'] = true_predictions
    

    # Plot historical data
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df[selected_province], label='Historical Data')
    plt.title('Historical Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    historical_plot_path = 'static/historical_prices.png'
    plt.savefig(historical_plot_path)
    plt.close()

    # Plot actual vs predicted values for 2023
    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test[selected_province], label='Actual Data', linestyle='--', color='blue')
    plt.plot(test.index, test['Predictions'], label='Predicted Data', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    predicted_plot_path = 'static/actual_vs_predicted.png'
    plt.savefig(predicted_plot_path)
    plt.close()

    # Prepare the table for comparison
    comparison_df = test

    # Extract the Date column and convert the DataFrame to a list of dictionaries
    comparison_data = comparison_df.round(2).reset_index().to_dict(orient='records')

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test[selected_province], test['Predictions']))
    mae = mean_absolute_error(test[selected_province], test['Predictions'])
    mape = mean_absolute_percentage_error(test[selected_province], test['Predictions'])
    r_squared = r2_score(test[selected_province], test['Predictions'])



    return render_template(
        'index.html',
        historical_plot=historical_plot_path,
        predicted_plot=predicted_plot_path,
         comparison_df=comparison_data,
        columns=columns,
        selected_province=selected_province,
        rmse=rmse,  # Pass RMSE to the template
        mae=mae,
        mape=mape,
        r_squared=r_squared
    )

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    conn = connect_to_database()

    # SQL query to fetch data
    sql_query = "SELECT * FROM swinetb;"

    # Fetch data from the database
    data = fetch_data_from_database(conn, sql_query)

    # Convert the fetched data to a DataFrame
    df = pd.DataFrame(data, columns=['id','Date','PHILIPPINES','Cordillera','Abra','Apayao','Benguet','Ifugao','Kalinga','MountainProvince','REGIONI',
                                        'IlocosNorte','IlocosSur','LaUnion','Pangasinan','REGIONII','Batanes','Cagayan','Isabela','NuevaVizcaya','Quirino',
                                        'REGIONIII','Aurora','Bataan','Bulacan','NuevaEcija','Pampanga','Tarlac','Zambales','REGIONIVA','Batangas','Cavite','Laguna',
                                        'Quezon','Rizal','MIMAROPAREGION','Marinduque','OccidentalMindoro','OrientalMindoro','Palawan','Romblon','REGIONV','Albay',
                                        'CamarinesNorte','CamarinesSur','Catanduanes','Masbate','Sorsogon','REGIONVI','Aklan','Antique','Capiz','Guimaras','Iloilo',
                                        'NegrosOccidental','REGIONVII','Bohol','Cebu','NegrosOriental','Siquijor','REGIONVIII','Biliran','EasternSamar','Leyte',
                                        'NorthernSamar','Samar','SouthernLeyte','REGIONIX','ZamboangadelNorte','ZamboangadelSur','ZamboangaSibugay','ZamboangaCity',
                                        'REGIONX','Bukidnon','Camiguin','LanaodelNorte','MisamisOccidental','MisamisOriental','REGIONXI','DavaodeOro','DavaodelNorte',
                                        'DavaodelSur','DavaoOccidental','DavaoOriental','CityofDavao','REGIONXII','Cotabato','Sarangani','SouthCotabato','SultanKudarat',
                                        'REGIONXIII','AgusandelNorte','AgusandelSur','SurigaodelNorte','SurigaodelSur','MUSLIMMINDANAO','Basilan','Maguindanao','Sulu'])

    # Get column names for dropdown options
    columns = df.columns[2:]  # Exclude 'id' and 'Date'

    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.drop(columns=['id'], inplace=True)  # Drop the 'id' column
    df.sort_index(inplace=True)

    # Convert non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with the mean
    df.fillna(df.mean(), inplace=True)

    # Default values for the GET request
    selected_province = 'PHILIPPINES'
    selected_year = '2024'
    predictions_df = get_prediction_pivot(conn, selected_province, int(selected_year))
    predicted_plot_path, prediction_table = None, None


    if request.method == 'POST':
        selected_province = request.form['province']
        selected_year = request.form['year']

    # Assuming you have the necessary functions like create_sequences, make_predictions_and_update_df, and store_predictions_in_database
    ph_df = pd.DataFrame(df, columns=[selected_province])
    model_filename = f'model/lstm_model_{selected_province.lower()}.h5'
    
    train = ph_df.iloc[:120]
    test = ph_df.iloc[120:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    val_data = scaled_train[96:]

    n_input = 18
    n_features = 1
    generator_train = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    generator_val = TimeseriesGenerator(val_data, val_data, length=n_input, batch_size=1)

    # Model Architechture
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer='l2'))

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

    try:
        model = load_model(model_filename)
    except:
        model.fit(generator_train, epochs=100, validation_data=generator_val, callbacks=[early_stop])
        # Save the model with the selected province in the filename
        model.save(model_filename)



    # Make predictions for 2024
    num_months_2024 = 12  # Number of months in 2024
    new_df_2024 = pd.DataFrame(df, columns=[selected_province])
    print(new_df_2024)
    scaled_train_2024 = scaler.transform(new_df_2024)
    current_batch_2024 = scaled_train_2024[-n_input:].reshape((1, n_input, n_features))


    updated_df_2024, df_2024 = make_predictions_and_update_df(
        model, current_batch_2024, n_input, n_features, num_months_2024, new_df_2024, scaler, '2024-01-01', scaled_train_2024, selected_province
    )

    store_predictions_in_database(df_2024, selected_province)



    # Make predictions for 2025
    num_months_2025 = 12  # Number of months in 2025
    updated_df_2025 = pd.DataFrame(updated_df_2024, columns=[selected_province])
    scaled_train_2025 = scaler.transform(updated_df_2025)
    current_batch_2025 = scaled_train_2025[-n_input:].reshape((1, n_input, n_features))


    updated_df_2025, df_2025 = make_predictions_and_update_df(
        model, current_batch_2025, n_input, n_features, num_months_2025, updated_df_2025, scaler, '2025-01-01', scaled_train_2025, selected_province
    )

    store_predictions_in_database(df_2025, selected_province)




    # Make predictions for 2026
    num_months_2026 = 12  # Number of months in 2026
    updated_df_2026 = pd.DataFrame(updated_df_2025, columns=[selected_province])
    scaled_train_2026 = scaler.transform(updated_df_2026)
    current_batch_2026 = scaled_train_2026[-n_input:].reshape((1, n_input, n_features))


    updated_df_2026, df_2026 = make_predictions_and_update_df(
        model, current_batch_2026, n_input, n_features, num_months_2026, updated_df_2026, scaler, '2026-01-01', scaled_train_2026, selected_province
    )

    store_predictions_in_database(df_2026, selected_province)

    

    # Make predictions for 2027
    num_months_2027 = 12  # Number of months in 2027
    updated_df_2027 = pd.DataFrame(updated_df_2026, columns=[selected_province])
    print(updated_df_2027[132:])
    scaled_train_2027 = scaler.transform(updated_df_2027)
    current_batch_2027 = scaled_train_2027[-n_input:].reshape((1, n_input, n_features))


    updated_df_2027, df_2027 = make_predictions_and_update_df(
        model, current_batch_2027, n_input, n_features, num_months_2027, updated_df_2027, scaler, '2027-01-01', scaled_train_2027, selected_province
    )

    store_predictions_in_database(df_2027, selected_province)




    # Fetch predictions for the selected province and year
    predictions_df = get_prediction_pivot(conn, selected_province, int(selected_year))


    if selected_year == '2024':
        df_plot = df_2024
    elif selected_year == '2025':
        df_plot = df_2025
    elif selected_year == '2026':
        df_plot = df_2026
    elif selected_year == '2027':
        df_plot = df_2027
    else:
        # Handle other years if needed
        df_plot = pd.DataFrame()

    # Plot and table for predictions based on the selected year
    predicted_plot_path, prediction_table = plot_and_table(df_plot, selected_province, selected_year)

    
    return render_template(
        'predictions.html',
        columns=columns,
        selected_province=selected_province,
        selected_year=selected_year,
        predicted_plot_path=predicted_plot_path,
        prediction_table=predictions_df.to_dict(orient='records'),
        prediction_pivot=predictions_df
    )


# Your existing app run code

if __name__ == '__main__':
    app.run(debug=True)