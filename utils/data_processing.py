import pandas as pd
import numpy as np

def resample_sensor_data(df, hospital_discharge):
    '''
    Resamples Healthdot data to hourly values (right-sided resampling). Also add a NightTime feature.
    Accounts for HR and RR being 0 if missing instead of NaN. Also applies algorithm to ensure data corresponds to
    wearing the sensor. Considers Healthdot data from 1 hour after hospital discharge onwards.

    :param df: dataframe with Healthdot data, with at least following columns: Time,Respirationrate,Activity,Pulserate
    :param hospital_discharge: the datetime of hospital discharge
    :return: resampled dataframe that contains RR, HR, Activity, and NightTime.
    '''

    # set NaNs if Pulserate or Respirationrate are 0, since 0 indicate missing in Healthdot data
    df.loc[df['Pulserate'] == 0, 'Pulserate'] = np.nan
    df.loc[df['Respirationrate'] == 0, 'Respirationrate'] = np.nan

    # format time array
    ts = pd.to_datetime(df['Time'], dayfirst=True).values
    ts = np.asarray(pd.Series(ts).dt.tz_localize(tz='UTC'))

    # store Healthdot data with time array in dataframe
    df = pd.DataFrame({'hr': df['Pulserate'].values, 'rr': df['Respirationrate'].values,
                       'act': df['Activity'].values}, index=ts)

    # resample hourly
    df = df.resample('1H', label='right').mean()

    # extract nighttime values and add to dataframe
    ts_ = np.asarray(pd.Series(df.index).dt.tz_convert(tz='Europe/Amsterdam'))
    df_nighttime = pd.DataFrame({'night': np.nan}, index=ts_)
    df['nighttime'] = df_nighttime.index.hour.map(nighttime)

    return df

def nighttime(hour):
    '''
    function that extracts a nighttime value for an hour

    :param hour: any integer between 0 and 23
    :return: 0, 0.33, 0.67 or 1; 1 is night, 0 is day, in between is transition period.
    '''
    if 10 <= hour <= 22:
        return 0
    elif 1 <= hour <= 7:
        return 1
    elif 22 < hour <= 24 or 0 <= hour < 1:
        # Linearly increase from 0 to 1 between 22 and 24
        return (hour - 22) / 3 if hour >= 23 else (hour + 2) / 3
    elif 7 < hour <= 9:
        # Linearly decrease from 1 to 0 between 7 and 10
        return 1 - ((hour - 7) / 3)
    return 0

def get_timecarrier(study_id='', hosp_discharge=pd.NaT, datetime_deterioration=pd.NaT):
    '''
    Extracts all decision moments in UTC timezone, corresponding to 7AM and 7PM in Amsterdam timezone.
    All decision moments are at least 1 hour after hospital discharge. No decision moments are extracted at or after
    time of deterioration

    :param study_id: study id to extract from
    :param hosp_discharge: datetime of hospital discharge
    :param datetime_deterioration: first datetime of deterioration. Could also be the datetime of early stopping or of a
    planned hospital visit
    :return: array of decision moments in UTC timezone corresponding to 7AM and 7PM in Amsterdam timezone
    '''

    # get the very first decision moment, which is either 7AM or 7PM, which is the closest.
    start_dt = hosp_discharge + np.timedelta64(1, 'h') # start from 1h after hospital discharge, to account for time to get home.

    # convert start_dt to amsterdam tz, as the decision moments are 7AM and 7PM following amsterdam tz
    start_dt_NL = pd.Series(start_dt).dt.tz_convert(tz='Europe/Amsterdam')

    # calculate the hours in UTC that correspond to 7AM and 7PM according to Amsterdam tz.
    utc_offsets = start_dt_NL.apply(lambda dt: dt.utcoffset().total_seconds() / 3600)
    tz_difference = utc_offsets.values[0]
    am7 = int(7 - tz_difference)
    pm7 = int(19 - tz_difference)

    # get the first decision moment
    current_hour = start_dt.hour
    if current_hour < am7: # if current hour is before 7AM on the same day, then 7AM will be first decision moment
        first_dt = start_dt.replace(hour=am7, minute=0, second=0, microsecond=0)
    elif current_hour >= pm7:  # if current hour is at or after 7PM same day, than 7AM of next day is first decision moment
        first_dt = start_dt.replace(hour=am7, minute=0, second=0, microsecond=0) + np.timedelta64(1, 'D')
    else: # if current hour is in between 7AM and 7PM, then 7PM of the same day is first decision moment
        first_dt = start_dt.replace(hour=pm7, minute=0, second=0, microsecond=0)

    # save first decision moment and new ones (12h later) until end_datetime is reached. No decision moments within 1
    # hour of the end datetime
    ts = [first_dt]
    while ts[-1] + np.timedelta64(12, 'h') <= datetime_deterioration - np.timedelta64(1, 'h'):

        # add new decision moment 12h later
        ts_add = ts[-1] + np.timedelta64(12 * 60 * 60, 's')

        # check whether there has not been a winter/summer time shift in between.
        # Else, change the UTC-hours that correspond to 7AM and 7PM in Amsterdam tz.
        ts_tzinfo = ts_add.tz_convert(tz='Europe/Amsterdam')
        tz_difference_new = ts_tzinfo.tzinfo._utcoffset.seconds / 3600
        if tz_difference_new != tz_difference:
            change = int(tz_difference_new - tz_difference)
            tz_difference = tz_difference_new
            ts_add = ts[-1] + np.timedelta64((12 - change) * 60 * 60, 's')

        # append new decision moment to list of decision moments
        ts.append(ts_add)

    # return array of decision moments in UTC timezone
    return ts


def label_decision_moments(timecarrier, deterioration, study_id):
    '''
    function to label all decision moments (1: postive, 0: neutral, -1: negative).
    If deteriorated (unplanned readmission, mortality or ed visit):
        last 2 decision moments are positive
        the 12 decision moments before the positive ones are neutral (should be skipped in model assessment)
        the remaining decision moments: negative

    if stopped:
        all decision moments are neutral (should be ignored in assessment)

    if planned:
        the 14 decision moments before event: neutral (should be ignored in model assessment)

    if not deteriorated:
        all decision moments are negative

    :param timecarrier: array of decision moments
    :param deterioration: dictionary with the keys ts and type, indicating datetime and type of deterioration
    :return:
    '''

    # get deterioration type
    type = deterioration['type']

    # assign labels
    labels = np.ones(len(timecarrier)) * -1
    if type == 'unplanned':
        timecarrier_mask_neutral = np.asarray(timecarrier) >= (deterioration['ts'] - np.timedelta64(7, 'D'))
        labels[timecarrier_mask_neutral] = 0

        timecarrier_mask_positive = np.asarray(timecarrier) >= (deterioration['ts'] - np.timedelta64(1, 'D'))
        labels[timecarrier_mask_positive] = 1

    elif type in {'planned'}:
        timecarrier_mask_neutral = np.asarray(timecarrier) >= (deterioration['ts'] - np.timedelta64(7, 'D'))
        labels[timecarrier_mask_neutral] = 0

    return pd.DataFrame({'y': labels, 'study_id': study_id}, index=timecarrier)
