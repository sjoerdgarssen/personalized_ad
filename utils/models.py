import numpy as np
import pandas as pd

class TemporalAD(object):
    # Algorithm which uses a list of models to predict test sample and uses weighted average of the predictions of each
    # of the listed models

    def __init__(self, models=None, importance_weight=1.5):
        '''

        :param models: list of models. Should be at least 1 model. Models should work according to sklearn style.
        :param importance_weight: the weight factor used for weighted average. Larger than 1: more weight to newer model
        '''
        self.models = models # list of models
        self.importance_weight = importance_weight # how much more important the model trained at t is than trained at t-1


    def predict(self, X, y=None):
        '''

        :param X: numpy array or pandas dataframe with input data (test samples).
        :param y: ignored, but included to for sklearn functionality.
        :return: float
        '''

        # initiate list to store scores for each test sample stored in X
        scores = []

        # loop over all samples in X
        for i in range(X.shape[0]):

            # loop over each model in the model list and store their predictions in list.
            predictions_ = []
            for model in self.models:

                # Predict on the test data. Account for different formats of X by try except.
                try:
                    prediction = model.score_samples(X.iloc[[i]])
                except:
                    prediction = model.score_samples(X[i].reshape(1, -1))

                # score_samples functions of sklearn LocalOutlierFactor and Isolation Forest: lower score --> more
                # abnormal. Change this such that higher score --> abnormal by multiplying by -1.
                prediction *= -1

                #store prediction for model
                predictions_.append(prediction[0])

            # Convert predictions to numpy array for easier use
            predictions = np.asarray(predictions_)

            # weighted average
            weights = [(1/self.importance_weight) ** i for i in range(len(self.models))]
            weights = list(reversed(weights))

            # Normalize weights
            weights /= np.sum(weights)

            # Compute weighted average and store in scores list
            score = np.average(predictions, axis=0, weights=weights)
            scores.append(score)

        return np.asarray(scores)

    def score_samples(self, X, y=None):
        return self.predict(X, y)

    def fit(self, X, y=None, sample_weight=None):
        self.feature_names = X.columns.values
        pass

    def transform(self, X, y=None):
        pass

    def get_features_names_out(self):
        return self.feature_names

class ShapWrapper:
    # Converts TemporalOCC to be seen as 1 model, despite using multiple models internally. This functionality is
    # needed to calculate shap values using shap.SamplingExplainer()

    def __init__(self, model, feature_names=None, df_type=None):
        '''

        :param model: TemporalOCC model
        :param feature_names: list of feature names
        :param df_type: boolean to indicate whether data input is dataframe
        '''
        self.df_type = df_type
        self.model = model
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # model is already fitted in TemporalOCC, but function is needed for sklearn functionality
        return self

    def predict(self, X):
        '''
        :param X: dataframe or numpy array of test set
        :return: scores
        '''

        if self.df_type:
            X = pd.DataFrame(data=X, columns=self.feature_names)
        return self.model.score_samples(X)
