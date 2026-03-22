import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
import shap
from home_deterioration.dummy.utils.models import *
from home_deterioration.dummy.utils.data_processing import *

def apply_algorithms(study_id, shap=False, discharge_date=pd.NaT, deteriorated={}, df_hd=pd.DataFrame()):
    '''
    function to apply isolation forest, local outlier factor and rews to a study subject.

    :param study_id: study id string
    :param shap: boolean, whether to calculate shap values or not
    :param discharge_date: datetime of hospital discharge
    :param deteriorated: dictionary with keys 'ts': datetime of event in UTC, 'type': 'unplanned', 'stopped', 'planned',
    'not_deteriorated'
    :param df_hd: loaded hd dataframe (raw)
    :return:
    '''

    # resample and engineer hd data
    df = resample_sensor_data(df_hd, hospital_discharge=discharge_date)

    # get timecarrier
    timecarrier = get_timecarrier(study_id=study_id, hosp_discharge=discharge_date,
                                  datetime_deterioration=deteriorated['ts'])

    # get abnormality scores of LOF and IF
    scores = apply_personalized_ad(timecarrier=timecarrier, df=df, shap=shap)

    # get rews scores
    rews_ = rews(timecarrier=timecarrier, df=df_hd)

    # join rews and scores
    df = scores.join(rews_)

    # get labels for decision moments in timecarrier and join to dataframe with rews and scores
    labels = label_decision_moments(timecarrier=timecarrier, deterioration=deteriorated, study_id=study_id)
    df = df.join(labels)
    df.dropna(inplace=True)
    if df.empty:
        return None

    return df

def apply_personalized_ad(timecarrier, df, rs=np.random.RandomState(1), shap=False):
    '''
    function that calculates abnormality scores for each decision moment using Local Outlier Factor or Isolation Forest.
    Predictions are made in a temporal way, in which up to 4 LOF or IF models of previous decision moments are re-used,
    in addition to a new model. Their predictions are merged into one prediction for each hour in a test set using
    weighted average. Also possible to extract shap values to indicate which feature contributed most to an abnormality
    score at a decision moment. Each model is based on HR, RR, Act and NightTime features.

    :param timecarrier: array with decision moments
    :param df: dataframe with resampled Healthdot data for one patient
    :param rs: randomstate numpy
    :param shap: boolean; whether to calculate shap or not
    :return: dataframe with as columns 'if' and 'lof' to indicate their scores. If shap==True, then more columns are
    used to store shap values of each feature.
    '''

    try:
        # make empty dataframe to store scores
        df_output = pd.DataFrame(index=timecarrier)

        # to store local outlier factor and isolation forest models
        loflist = []
        iflist = []

        # loop over decision moments
        for decision_moment in timecarrier:

            # split data into train and test
            train, test = get_train_test(df, decision_moment)

            if test.shape[0] == 0:  # if no test data, go to next decision moment
                continue

            # check quality of train data. If insufficient, store NaNs as scores for current decision moment and go to
            # the next decision moment. If True, then quality is insufficient.
            if quality_check_train(train):
                df_output.loc[decision_moment, 'lof'] = np.nan  # local outlier factor
                df_output.loc[decision_moment, 'if'] = np.nan  # isolation forest
                continue

            # Local Outlier Factor
            loflist = fit_lof(train, loflist)
            predict_models = TemporalAD(importance_weight=1.5, models=loflist)
            wavg_risk_score = predict_models.predict(X=test)
            df_output.loc[decision_moment, 'lof'] = wavg_risk_score.max()

            # if shap values should be calculated
            if shap:

                # get feature names
                features_out = loflist[-1].named_steps['standardize'].get_feature_names_out()

                # calculate shap
                shap_values = calculate_shap(temporal_models=predict_models, features_names=features_out,
                                             risk_scores=wavg_risk_score, train=train.loc[:, features_out],
                                             test=test.loc[:, features_out])

                # store shap values
                for ix, col in enumerate(test.columns):
                    df_output.loc[decision_moment, 'lof_' + col] = shap_values[ix]

            # Isolation Forest
            iflist = fit_if(train, iflist, random_state=rs)
            predict_models = TemporalAD(importance_weight=1.5, models=iflist)
            wavg_risk_score = predict_models.predict(X=np.asarray(test))
            df_output.loc[decision_moment, 'if'] = wavg_risk_score.max()

            # if shap values should be calculated
            if shap:

                # get feature names
                features_out = train.columns

                # calculate shap
                shap_values = calculate_shap(temporal_models=predict_models, features_names=features_out,
                                             risk_scores=wavg_risk_score, train=train.values, test=test.values)

                # store shap
                for ix, col in enumerate(test.columns):
                    df_output.loc[decision_moment, 'if_' + col] = shap_values[ix]

        return df_output

    except Exception as e:
        print(f'An error occurred: {e}')
        return None


def get_train_test(df, decision_moment, ffill=True):
    '''
    function to extract train and test dataframes for a decision moment. Test data is the last 12h, train data
    everything before.

    :param df: healthdot dataframe
    :param decision_moment: timepoint of decision moment
    :return: train dataframe and test dataframe
    '''

    # test
    test_mask = (df.index <= decision_moment) & (df.index > decision_moment - np.timedelta64(12, 'h'))
    test = df.loc[test_mask, :]
    if ffill:
        test = test.fillna(method='ffill')
        test = test.dropna(axis=0)

    # train
    train_mask = df.index <= decision_moment - np.timedelta64(12, 'h')
    train = df[train_mask]
    train = train.dropna(axis=0)

    return train, test

def quality_check_train(train):
    '''
    function that checks whether train dataframe has hourly coverage and has enough rows to train models.

    :param train: dataframe with train data
    :return: boolean to indicate whether current decision moment should be passed
    '''

    # if train data has less than 12 rows, save NaNs as scores for current decision moment. Train set too small.
    train.dropna(inplace=True)
    if train.shape[0] < 12:
        return True

    # check whether train data has hourly coverage. If not, then save NaNs as scores for current decision moment
    covered = check_hourly_coverage(train)
    if not covered:
        return True

    return False


def check_hourly_coverage(df):
    '''
    Function to check whether each hour of the day has coverage in healthdot data, with allowance of 2 hours deviation
    :param df: healthdot data
    :return: True if covered, False if not.
    '''

    # drop NaNs
    df_ = df.dropna()

    # if no data left, return False
    if df_.shape[0] == 0:
        return False

    # else check whether each hour is covered
    else:
        hours = df_.index.hour
        for hour in range(24):
            covered = False
            # Check the hour itself and the ±2 deviation, considering wrap-around using modulo 24
            for h in range(hour - 2, hour + 2):
                if (h % 24) in hours:
                    covered = True
                    break
            if not covered:
                return False

        return True

def fit_lof(train, loflist):
    '''
    function that fits LOF and adds model to loflist.

    :param train: train dataframe
    :param loflist: list with earlier trained lof models
    :return: loflist
    '''

    # calculate the number of neighbours to use in LOF, as square root of nr of rows in train dataframe (rounded up)
    neighbours = int(np.ceil(np.sqrt(train.shape[0])))

    # set preprocessing technique. NightTime should not be preprocessed
    columns_to_0_1 = [col for col in train.columns if 'ight' not in col]
    preprocessor = ColumnTransformer(
        transformers=[
            ('standardize_iqr', RobustScaler(quantile_range=(25, 75)), columns_to_0_1)],
        remainder='passthrough', verbose_feature_names_out=False
    )

    # set pipeline
    pipeline = Pipeline([
        ('standardize', preprocessor),
        ('classification',
         LocalOutlierFactor(novelty=True, n_neighbors=neighbours))]
    )

    # fit pipeline
    lof = pipeline.fit(X=train)
    loflist.append(lof)
    if len(loflist) > 5:
        loflist = loflist[-5:]

    return loflist

def fit_if(train, iflist, random_state):
    '''
    function that fits IF and adds model to iflist.

    :param train: train dataframe
    :param iflist: list with earlier trained if models
    :param random_state: numpy random state
    :return: iflist
    '''

    # set and fit model
    isofor = IsolationForest(random_state=random_state, max_samples=1.0, n_estimators=100, max_features=1.0)
    isofor = isofor.fit(X=np.asarray(train))
    iflist.append(isofor)
    if len(iflist) > 5:
        iflist = iflist[-5:]

    return iflist

def calculate_shap(temporal_models, features_names, risk_scores, train, test):
    '''
    function that calculated the shapvalues to indicate which feature contributed most to outcome value

    :param temporal_models: output of TemporalOCC
    :param features_names: feature names in the same order as used in the model
    :param risk_scores: resulting risk scores for each hour
    :param train: train dataframe
    :param test: test dataframe
    :return: array of shap values
    '''

    # get most anomalous index of risk scores of test set, as that timepoint should be assessed on shap values
    most_anomalous_index = np.argmax(risk_scores)

    # calculate shap values
    model = ShapWrapper(temporal_models, feature_names=features_names, df_type=isinstance(train, pd.DataFrame))
    explainer = shap.SamplingExplainer(model.predict, train)

    # return shap values
    if isinstance(test, pd.DataFrame):
        shap_values = explainer.shap_values(X=test.iloc[most_anomalous_index, :], nsamples=test.shape[1] * 100)
    else:
        shap_values = explainer.shap_values(X=test[most_anomalous_index, :], nsamples=test.shape[1] * 100)

    return shap_values

def rews(timecarrier, df):
    '''
    Calculates the rews on the data of the 12h preceding decision moments.

    :param timecarrier: array with decision moments
    :param df: dataframe
    :return: rews score for each decision moment
    '''

    try:
        # make empty dataframe to store scores
        df_output = pd.DataFrame(index=timecarrier)

        # set NaNs if Pulserate or Respirationrate are 0, since 0 indicate missing in Healthdot data
        df.loc[df['Pulserate'] == 0, 'Pulserate'] = np.nan
        df.loc[df['Respirationrate'] == 0, 'Respirationrate'] = np.nan

        # calculate rews points of hr
        df['REWS_HR'] = pd.cut(df['Pulserate'], bins=[1, 39, 50, 100, 110, 130, np.inf], labels=[2, 1, 0, 1, 2, 3],
                               ordered=False)
        df['REWS_HR'] = pd.to_numeric(df['REWS_HR'])

        # calculate rews points of rr
        df['REWS_RR'] = pd.cut(df['Respirationrate'], bins=[1, 8, 14, 20, 30, np.inf], labels=[2, 0, 1, 2, 3],
                               ordered=False)
        df['REWS_RR'] = pd.to_numeric(df['REWS_RR'])

        # format time array
        ts = pd.to_datetime(df['Time'], dayfirst=True).values
        ts = np.asarray(pd.Series(ts).dt.tz_localize(tz='UTC'))

        # store Healthdot data with time array in dataframe
        df = pd.DataFrame({'rews_hr': df['REWS_HR'].values, 'rews_rr': df['REWS_RR'].values}, index=ts)

        # loop over decision moments
        for decision_moment in timecarrier:

            # get test data
            _, test = get_train_test(df=df, ffill=False, decision_moment=decision_moment)

            # If only HR or RR was observed, use the least observation for the other to be able to calculate a score.
            # if both are not observed, then no rews is calculated.
            mask = (np.isnan(test['rews_hr'])) & (np.isnan(test['rews_rr']))
            test = test.fillna(method='ffill')
            test['rews'] = test['rews_hr'] + test['rews_rr']
            test['rews'][mask] = np.nan

            # resample by taking the mean over past hour
            test = test.resample('1H', label='right').mean()
            test = test.dropna(axis=0)

            # store rews
            df_output.loc[decision_moment, 'rews'] = test['rews'].max()

        return df_output

    except Exception as e:
        print(f'An error occurred: {e}')
        return None
