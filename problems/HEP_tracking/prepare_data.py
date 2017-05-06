import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit


def get_cv(y_train_array, test_size):
    unique_event_ids = np.unique(y_train_array[:, 0])
    event_cv = ShuffleSplit(
    	n_splits=1, test_size=test_size, random_state=57)
    for train_event_is, test_event_is in event_cv.split(unique_event_ids):
        train_is = np.where(
        	np.in1d(y_train_array[:, 0], unique_event_ids[train_event_is]))
        test_is = np.where(
        	np.in1d(y_train_array[:, 0], unique_event_ids[test_event_is]))
        yield train_is, test_is


if __name__ == '__main__':
	data_df = pd.read_csv(
		'data/raw/all.csv')
	print data_df.head()
	print data_df.values[:, 0]

	data_df = data_df[data_df['event_id'] < 20000]
	cv = get_cv(data_df.values, 0.75)
	train_is, test_is = list(cv)[0]
	public_train_df = data_df.iloc[train_is]
	print 'public train n_sample =', public_train_df.shape[0]
	print 'public train n_event =', np.unique(public_train_df['event_id']).shape[0]
	private_df = data_df.iloc[test_is]
	print 'private train n_sample =', private_df.shape[0]
	print 'private train n_event =', np.unique(private_df['event_id']).shape[0]
	
	cv = get_cv(private_df.values, 0.33)
	train_is, test_is = list(cv)[0]
	train_df = private_df.iloc[train_is]
	print 'train n_sample =', train_df.shape[0]
	print 'train n_event =', np.unique(train_df['event_id']).shape[0]
	test_df = private_df.iloc[test_is]
	print 'test n_sample =', test_df.shape[0]
	print 'test n_event =', np.unique(test_df['event_id']).shape[0]

	public_train_df.to_csv('starting_kit/public_train.csv', index=False)
	train_df.to_csv('data/private/train.csv', index=False)
	test_df.to_csv('data/private/test.csv', index=False)