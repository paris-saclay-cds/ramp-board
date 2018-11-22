import numpy as np

from .model import Team
from .model import Event
from .model import EventTeam
from .model import combine_predictions_list

__all__ = [
    'get_active_user_event_team',
    'get_n_team_members',
    'get_n_user_teams',
    'get_team_members',
    'get_user_teams',
    'get_user_event_teams',
    'get_next_best_single_fold',
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
