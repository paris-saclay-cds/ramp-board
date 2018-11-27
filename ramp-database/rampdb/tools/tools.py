import numpy as np

from ..model import Team
from ..model import Event
from ..model import EventTeam

__all__ = [
    'get_active_user_event_team',
    'get_n_team_members',
    'get_n_user_teams',
    'get_team_members',
    'get_user_teams',
    'get_user_event_teams',
    'get_next_best_single_fold',
    'combine_predictions_list',
]


def get_active_user_event_team(event, user):
    # There should always be an active user team, if not, throw an exception
    # The current code works only if each user admins a single team.
    event_team = EventTeam.query.filter_by(
        event=event, team=user.admined_teams[0]).one_or_none()

    return event_team


def get_team_members(team):
    return team.admin


def get_n_team_members(team):
    return len(list(get_team_members(team)))


def get_user_teams(user):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    return Team.query.filter_by(name=user.name).one()


def get_user_event_teams(event_name, user_name):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=user_name).one()

    return EventTeam.query.filter_by(event=event, team=team).one_or_none()


def get_n_user_teams(user):
    return len(get_user_teams(user))


def get_next_best_single_fold(event, predictions_list, ground_truths,
                              best_index_list):
    """.

    Find the model that minimizes the score if added to
    predictions_list[best_index_list] using event.official_score_function.
    If there is no model improving the input
    combination, the input best_index_list is returned. Otherwise the best
    model is added to the list. We could also return the combined prediction
    (for efficiency, so the combination would not have to be done each time;
    right now the algo is quadratic), but I don't think any meaningful
    rule will be associative, in which case we should redo the combination from
    scratch each time the set changes. Since now combination = mean, we could
    maintain the sum and the number of models, but it would be a bit bulky.
    We'll see how this evolves.

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a model
        on the same (cross-validation valid) data points.
    ground_truths : instance of Predictions
        The ground truth.
    best_index_list : list of integers
        Indices of the current best model.

    Returns
    -------
    best_index_list : list of integers
        Indices of the models in the new combination. If the same as input,
        no models wer found improving the score.
    """
    best_predictions = combine_predictions_list(
        predictions_list, index_list=best_index_list)
    best_score = event.official_score_function(
        ground_truths, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles
    r = np.arange(len(predictions_list))
    # Randomization doesn't matter, only in case of exact equality.
    # np.random.shuffle(r)
    # print r
    for i in r:
        combined_predictions = combine_predictions_list(
            predictions_list, index_list=np.append(best_index_list, i))
        new_score = event.official_score_function(
            ground_truths, combined_predictions)
        is_lower_the_better = event.official_score_type.is_lower_the_better
        if (is_lower_the_better and new_score < best_score) or\
                (not is_lower_the_better and new_score > best_score):
            best_predictions = combined_predictions
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index), best_score
    else:
        return best_index_list, best_score


def combine_predictions_list(predictions_list, index_list=None):
    """Combine predictions in predictions_list[index_list].

    By taking the mean of their get_combineable_predictions views.

    E.g. for regression it is the actual
    predictions, and for classification it is the probability array (which
    should be calibrated if we want the best performance). Called both for
    combining one submission on cv folds (a single model that is trained on
    different folds) and several models on a single fold.
    Called by
    _get_bagging_score : which combines bags of the same model, trained on
        different folds, on the heldout test set
    _get_cv_bagging_score : which combines cv-bags of the same model, trained
        on different folds, on the training set
    get_next_best_single_fold : which does one step of the greedy forward
        selection (of different models) on a single fold
    _get_combined_predictions_single_fold : which does the full loop of greedy
        forward selection (of different models), until improvement, on a single
        fold
    _get_combined_test_predictions_single_fold : which computes the combination
        (constructed on the cv valid set) on the holdout test set, on a single
        fold
    _get_combined_test_predictions : which combines the foldwise combined
        and foldwise best test predictions into a single megacombination

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a given model
        on the same data points.
    index_list : None | list of integers
        The subset of predictions to be combined. If None, the full set is
        combined.

    Returns
    -------
    combined_predictions : instance of Predictions
        A predictions instance containing the combined (averaged) predictions.
    """
    Predictions = type(predictions_list[0])
    combined_predictions = Predictions.combine(predictions_list, index_list)
    return combined_predictions


def _get_score_cv_bags(event, score_type, predictions_list, ground_truths,
                       test_is_list=None):
    """
    Computes the bagged score of the predictions in predictions_list.

    Called by Submission.compute_valid_score_cv_bag and
    db_tools.compute_contributivity.

    Parameters
    ----------
    event : instance of Event
        Needed for the type of y_comb and
    predictions_list : list of instances of Predictions
    ground_truths : instance of Predictions
    test_is_list : list of integers
        Indices of points that should be bagged in each prediction. If None,
        the full prediction vectors will be bagged.
    Returns
    -------
    score_cv_bags : instance of Score ()
    """
    if test_is_list is None:  # we combine the full list
        test_is_list = [range(len(predictions.y_pred))
                        for predictions in predictions_list]

    y_comb = np.array(
        [event.Predictions(n_samples=len(ground_truths.y_pred))
         for _ in predictions_list])
    score_cv_bags = []
    for i, test_is in enumerate(test_is_list):
        y_comb[i].set_valid_in_train(predictions_list[i], test_is)
        combined_predictions = combine_predictions_list(y_comb[:i + 1])
        valid_indexes = combined_predictions.valid_indexes
        score_cv_bags.append(score_type.score_function(
            ground_truths, combined_predictions, valid_indexes))
        # XXX maybe use masked arrays rather than passing valid_indexes
    return combined_predictions, score_cv_bags
