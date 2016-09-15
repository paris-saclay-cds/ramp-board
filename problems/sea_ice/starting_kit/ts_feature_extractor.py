import numpy as np
import xarray as xr


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the vector of input variables at time t. Spatial variables will
        be averaged along lat and lon coordinates."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = np.arange(X_ds.attrs['n_burn_in'], len(X_ds['time']))
        # We convert the Dataset into a 4D DataArray
        X_xr = X_ds.to_array()
        # We compute the mean over the lat and lon axes
        mean_xr = np.mean(X_xr, axis=(2, 3))
        # We convert it into numpy array, transpose, and slice the valid range
        X_array = mean_xr.values.T[valid_range]
        return X_array