import numpy as np
from importlib import import_module


def train_submission(module_path, X_array, y_array, train_is):
    clusterer = import_module('.clusterer', module_path)
    ctr = clusterer.Clusterer()
    ctr.fit(X_array[train_is], y_array[train_is])
    return ctr


def test_submission(trained_model, X_array, test_is):
    ctr = trained_model
    X = X_array[test_is]
    unique_event_ids = np.unique(X[:, 0])
    cluster_ids = np.empty(len(X), dtype='int')

    for event_id in unique_event_ids:
        event_indices = (X[:, 0] == event_id)
        # select an event and drop event ids
        X_event = X[event_indices][:, 1:]
        cluster_ids[event_indices] = ctr.predict_single_event(X_event)

    return np.stack((X[:, 0], cluster_ids), axis=-1).astype(dtype='int')
