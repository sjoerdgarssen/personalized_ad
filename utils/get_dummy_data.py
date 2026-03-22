import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_dataframes(num_days=5, freq_minutes=5, id=1):
    '''
    functions that generates dummy sensor data for a dummy patient
    :param num_days: number of days to generate data for
    :param freq_minutes: how many times it takes for next measurement
    :param id: study id
    :return: dataframe with dummy data
    '''
    # Generate time range
    # Start time is set to January 1, 2025
    start_time = datetime(2025, 1, 1)
    # End time is calculated by adding the specified number of days
    end_time = start_time + timedelta(days=num_days)
    # Create a time range with the specified frequency in UTC timezone
    time_range = pd.date_range(start=start_time, end=end_time, freq=f'{freq_minutes}T', tz='UTC')

    # Generate data
    # Number of rows corresponds to the length of the time range
    num_rows = len(time_range)
    # Generate a list of patient IDs
    patient_id = [id] * num_rows
    # Generate random respiration rates between 8 and 35
    respiration_rate = np.random.randint(8, 35, size=num_rows)
    # Generate random activity levels between 0 and 10
    activity = np.random.randint(0, 11, size=num_rows)
    # Generate random pulse rates between 40 and 140
    pulse_rate = np.random.randint(40, 140, size=num_rows)

    # Create a DataFrame with the generated data
    df = pd.DataFrame({
        'PatientId': patient_id,  # Patient ID column
        'Time': time_range,  # Time column
        'Respirationrate': respiration_rate,  # Respiration rate column
        'Activity': activity,  # Activity column
        'Pulserate': pulse_rate  # Pulse rate column
    })

    # Introduce missing rows
    # Randomly drop 30% of the rows to simulate missing data
    missing_rows = df.sample(frac=0.3, random_state=42).index
    df = df.drop(index=missing_rows)

    # Introduce missing values
    # Randomly set 15% of the remaining rows' Respirationrate and Pulserate to NaN
    missing_values = df.sample(frac=0.15, random_state=42).index
    df.loc[missing_values, ['Respirationrate', 'Pulserate']] = np.nan

    # Return the generated DataFrame
    return df
