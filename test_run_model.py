from databoard.specific import run_model, split_data
from databoard.generic import score
import test_model

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, skf = split_data()
    for valid_train_is, valid_test_is in skf:
        X_valid_train = X_train[valid_train_is]
        y_valid_train = y_train[valid_train_is]
        X_valid_test = X_train[valid_test_is]
        y_valid_test = y_train[valid_test_is]
        y_valid_pred, y_valid_score, y_test_pred, y_test_score = run_model(
            test_model, X_valid_train, y_valid_train, X_valid_test, X_test)

        print score(y_valid_score[:,1], y_valid_test)
        print score(y_test_score[:,1], y_test)
        # I couldn't figure out how to get just the first element of skf
        # train_is, test_is = next(skf)
        # TypeError: StratifiedShuffleSplit object is not an iterator
        # so for now it's ugly, pls find a better way
        #break  