import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit


def train_submission(module_path, X_df, y_array, train_is):
    # Preparing the training set
    X_train_df = X_df.iloc[train_is]
    y_train_array = y_array[train_is]

    # Feature extraction
    import feature_extractor
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_df, y_train_array)
    X_train_array = fe.transform(X_train_df)

    import regressor
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg


def test_submission(trained_model, X_df, test_is):

    # Preparing the test (or valid) set
    X_test_df = X_df.iloc[test_is]

    fe, reg = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_df)

    # Regression
    y_pred = reg.predict(X_test_array)
    return y_pred


data = pd.read_csv("public_train.csv")
X_df = data.drop(['log_PAX'], axis=1)
y_array = data['log_PAX'].values

skf = ShuffleSplit(y_array.shape[0], n_iter=2, test_size=0.2, random_state=61)
skf_is = list(skf)[0]
train_is, test_is = skf_is

trained_model = train_submission('.', X_df, y_array, train_is)
y_pred_array = test_submission(trained_model, X_df, test_is)
ground_truth_array = y_array[test_is]

score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print 'RMSE =', score