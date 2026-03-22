# This script is used to run a dummy experiment for personalized anomaly detection

# Import necessary libraries
from home_deterioration.dummy.utils.train_and_apply import *
from home_deterioration.dummy.utils.get_dummy_data import *

# Define types of events
events = ['not_deteriorated', 'unplanned', 'planned']

# Initialize an empty DataFrame to store the output
output_df = pd.DataFrame()

# Loop through 10 dummy patients
for study_idx in range(0,10):

    # get study ID
    study_id = f'S{study_idx}'

    # load dummy sensor data
    df_sensor = generate_dummy_dataframes(id=study_id)

    # load dummy discharge and deterioration dates
    discharge_date = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    event_nr = study_idx % 3
    deteriorated = {'ts': pd.Timestamp('2025-01-06 00:00:00', tz='UTC'), 'type': events[event_nr]}

    # calculate shap values if events was unplanned
    if events[event_nr] == 'unplanned':
        shap = True
    else:
        shap = False

    # apply algorithms
    df = apply_algorithms(study_id=study_id, shap=shap, discharge_date=discharge_date, deteriorated=deteriorated,
                          df_hd=df_sensor)

    # save the results
    output_df = pd.concat([output_df, df])
