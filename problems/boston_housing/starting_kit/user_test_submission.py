import pandas as pd
import numpy as np
from sklearn.cross_validation import ShuffleSplit
import regressor

target_column_name = 'medv'
# point it to your training file
filename = 'public_train.csv'

if __name__ == '__main__':
    df = pd.read_csv(filename)
    y = df[target_column_name].values
    X = df.drop(target_column_name, axis=1).values
    skf = ShuffleSplit(len(y), n_iter=2, test_size=0.5, random_state=61)
    for valid_train_is, valid_test_is in skf:
        X_valid_train = X[valid_train_is]
        y_valid_train = y[valid_train_is]
        X_valid_test = X[valid_test_is]
        y_valid_test = y[valid_test_is]
        reg = regressor.Regressor()
        reg.fit(X_valid_train, y_valid_train)
        y_valid_pred = reg.predict(X_valid_test)
        print 'rmse = ', np.sqrt(
            np.mean(np.square(y_valid_test - y_valid_pred)))
