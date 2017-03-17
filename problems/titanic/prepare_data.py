import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == '__main__':
    data_df = pd.read_csv('data/raw/all.csv')
    print data_df.head(1)
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=57)
    train_is, test_is = list(
        skf.split(data_df, data_df['Survived'].values))[0]
    train_df = data_df.iloc[train_is]
    test_df = data_df.iloc[test_is]

    train_df.to_csv('data/private/train.csv', index=False)
    test_df.to_csv('data/private/test.csv', index=False)
    train_df.to_csv('starting_kit/train.csv', index=False)
