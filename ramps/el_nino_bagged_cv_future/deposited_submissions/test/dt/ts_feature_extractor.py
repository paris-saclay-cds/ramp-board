import numpy as np
 
en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120


def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region at all time 
    points."""
    return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(
        dim=('lat', 'lon'))


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        """Combine two variables: the montly means corresponding to the month
        of the target and the current mean temperature in the El Nino 3.4
        region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = range(
            n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        enso = get_enso_mean(temperatures_xray['tas'])
        # reshape the vector into a table years as rows, months as columns
        enso_matrix = enso.values.reshape((-1, 12))
        count_matrix = np.ones(enso_matrix.shape)
        # compute cumulative means of columns (remember that you can only use
        # the past at each time point) and reshape it into a vector
        enso_monthly_mean = (enso_matrix.cumsum(axis=0) /
                             count_matrix.cumsum(axis=0)).ravel()
        # roll it backwards (6 months) so it corresponds to the month of the
        # target
        enso_monthly_mean_rolled = np.roll(enso_monthly_mean, n_lookahead - 12)
        # select valid range
        enso_monthly_mean_valid = enso_monthly_mean_rolled[valid_range]
        enso_valid = enso.values[valid_range]
        X = np.array([enso_valid, enso_monthly_mean_valid]).T
        return X
