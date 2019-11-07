import logging
import numpy as np

from rampwf.utils import get_score_cv_bags

from ..model import CVFold
from ..model import SubmissionSimilarity

from ._query import select_event_by_name
from ._query import select_submissions_by_state

logger = logging.getLogger('RAMP-DATABASE')


def compute_historical_contributivity(session, event_name):
    """Compute historical contributivities of an event using
       contributivities from blending and credits.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event associated to the submission.
    """
    submissions = select_submissions_by_state(
        session, event_name, state='scored')
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    for s in submissions:
        s.historical_contributivity = 0.0
    for s in submissions:
        s.historical_contributivity += s.contributivity
        similarities = session.query(SubmissionSimilarity).filter_by(
            type='target_credit', target_submission=s).all()
        if similarities:
            # if a target team enters several credits to a source submission
            # we only take the latest
            similarities.sort(key=lambda x: x.timestamp, reverse=True)
            processed_submissions = []
            historical_contributivity = s.historical_contributivity
            for ss in similarities:
                source_submission = ss.source_submission
                if source_submission not in processed_submissions:
                    partial_credit = historical_contributivity * ss.similarity
                    source_submission.historical_contributivity +=\
                        partial_credit
                    s.historical_contributivity -= partial_credit
                    processed_submissions.append(source_submission)
    session.commit()


def compute_contributivity(session, event_name):
    """Blend submissions of an event, compute combined score and
       contributivities.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event associated to the submission.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info('Combining models')

    event = select_event_by_name(session, event_name)
    score_type = event.get_official_score_type(session)
    submissions = select_submissions_by_state(
        session, event_name, state='scored')

    ground_truths_train = event.problem.ground_truths_train()
    ground_truths_test = event.problem.ground_truths_test()

    combined_predictions_list = []
    best_predictions_list = []
    combined_test_predictions_list = []
    best_test_predictions_list = []
    test_is_list = []

    for cv_fold in session.query(CVFold).filter_by(event=event).all():
        logger.info('{}'.format(cv_fold))
        ground_truths_valid = event.problem.ground_truths_valid(
            cv_fold.test_is)
        combined_predictions, best_predictions,\
            combined_test_predictions, best_test_predictions =\
            compute_contributivity_on_fold(
                session, cv_fold, ground_truths_valid)
        if combined_predictions is None:
            logger.info('No submissions to combine')
            return
        combined_predictions_list.append(combined_predictions)
        best_predictions_list.append(best_predictions)
        combined_test_predictions_list.append(combined_test_predictions)
        best_test_predictions_list.append(best_test_predictions)
        test_is_list.append(cv_fold.test_is)

    for submission in submissions:
        submission.set_contributivity()
    # if there are no predictions to combine, it crashed
    combined_predictions_list = [c for c in combined_predictions_list
                                 if c is not None]
    if len(combined_predictions_list) > 0:
        combined_predictions, scores = get_score_cv_bags(
            score_type, combined_predictions_list,
            ground_truths_train, test_is_list=test_is_list)
        logger.info('Combined combined valid score = {}'.format(scores))
        event.combined_combined_valid_score = float(scores[-1])
    else:
        event.combined_combined_valid_score = None

    best_predictions_list = [c for c in best_predictions_list
                             if c is not None]
    if len(best_predictions_list) > 0:
        _, scores = get_score_cv_bags(
            score_type, best_predictions_list,
            ground_truths_train, test_is_list=test_is_list)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_valid_score = float(scores[-1])
    else:
        event.combined_foldwise_valid_score = None

    combined_test_predictions_list = [c for c in combined_test_predictions_list
                                      if c is not None]
    if len(combined_test_predictions_list) > 0:
        combined_predictions, scores = get_score_cv_bags(
            score_type, combined_test_predictions_list, ground_truths_test)
        logger.info('Combined combined test score = {}'.format(scores))
        event.combined_combined_test_score = float(scores[-1])
    else:
        event.combined_combined_test_score = None

    best_test_predictions_list = [c for c in best_test_predictions_list
                                  if c is not None]
    if len(best_test_predictions_list) > 0:
        _, scores = get_score_cv_bags(
            score_type, best_test_predictions_list, ground_truths_test)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_test_score = float(scores[-1])
    else:
        event.combined_foldwise_test_score = None

    session.commit()


def compute_contributivity_on_fold(session, cv_fold, ground_truths_valid,
                                   start_time_stamp=None, end_time_stamp=None,
                                   force_ensemble=False, min_improvement=0.0):
    """Construct the best model combination on a single fold.

    We blend models on a fold using greedy forward selection with replacement,
    see reference below. We return the predictions of both the best model and
    combined (blended) model, for both the validation set and the test set.
    We set foldwise contributivity based on the integer weight in the enseble.

    Reference
    ---------
    `Greedy forward selection <
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/
    caruana.icml04.icdm06long.pdf>`_

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    cv_fold : pair of integer arrays
        The cv fold indices.
    ground_truths_valid : :class:`rampwf.prediction_types.BasePrediction`
        The validation ground truths.
    start_time_stamp : datetime or None, default is None
        Starting time stamp for submission selection.
    end_time_stamp : datetime or None, default is None
        Ending time stamp for submission selection.
    force_ensemble : bool, default is False
        To force include deleted models.
    min_improvement : float, default is 0.0
        The minimum improvement needed to continue the greedy loop.
    Returns
    -------
    combined_predictions : :class:`rampwf.prediction_types.BasePrediction`
        combined (blended) validation predictions
    best_predictions : :class:`rampwf.prediction_types.BasePrediction`
        validation predictions of the best model
    combined_test_predictions : \
        :class:`rampwf.prediction_types.BasePrediction`
        combined (blended) test predictions
    best_test_predictions : :class:`rampwf.prediction_types.BasePrediction`
        test predictions of the best model
    """
    # The submissions must have is_to_ensemble set to True. It is for
    # fogetting models. Users can also delete models in which case
    # we make is_valid false. We then only use these models if
    # force_ensemble is True.
    # We can further bag here which should be handled in config (or
    # ramp table.) Or we could bag in get_next_best_single_fold

    # this is the bottleneck
    selected_submissions_on_fold = [
        submission_on_fold for submission_on_fold in cv_fold.submissions
        if (submission_on_fold.submission.is_valid or force_ensemble) and
        submission_on_fold.submission.is_to_ensemble and
        submission_on_fold.submission.is_in_competition and
        submission_on_fold.state == 'scored' and
        submission_on_fold.submission.is_not_sandbox
    ]
    # reset
    for submission_on_fold in selected_submissions_on_fold:
        submission_on_fold.best = False
        submission_on_fold.contributivity = 0.0
    # select submissions in time interval
    if start_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp >=
            start_time_stamp
        ]
    if end_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp <=
            end_time_stamp
        ]

    if len(selected_submissions_on_fold) == 0:
        return None, None, None, None
    # TODO: maybe this can be simplified. Don't need to get down
    # to prediction level.
    predictions_list = [
        submission_on_fold.valid_predictions
        for submission_on_fold in selected_submissions_on_fold]
    Predictions = type(predictions_list[0])
    valid_scores = [
        submission_on_fold.official_score.valid_score
        for submission_on_fold in selected_submissions_on_fold]
    if cv_fold.event.get_official_score_type(session).is_lower_the_better:
        best_prediction_index = np.argmin(valid_scores)
    else:
        best_prediction_index = np.argmax(valid_scores)
    best_index_list = np.array([best_prediction_index])
    improvement = True
    while improvement and len(best_index_list) < cv_fold.event.max_n_ensemble:
        old_best_index_list = best_index_list
        best_index_list, score = get_next_best_single_fold(
            session, cv_fold.event, predictions_list, ground_truths_valid,
            best_index_list, min_improvement)
        improvement = len(best_index_list) != len(old_best_index_list)
        logger.info('\t{}: {}'.format(old_best_index_list, score))
    # set
    selected_submissions_on_fold[best_index_list[0]].best = True
    # we share a unit of 1. among the contributive submissions
    unit_contributivity = 1. / len(best_index_list)
    for i in best_index_list:
        selected_submissions_on_fold[i].contributivity +=\
            unit_contributivity
    combined_predictions = Predictions.combine(
        predictions_list, best_index_list)
    best_predictions = predictions_list[best_index_list[0]]

    test_predictions_list = [
        submission_on_fold.test_predictions
        for submission_on_fold in selected_submissions_on_fold
    ]
    if any(test_predictions_list) is None:
        logger.error("Can't compute combined test score," +
                     " some submissions are untested.")
        combined_test_predictions = None
        best_test_predictions = None
    else:
        combined_test_predictions = Predictions.combine(
            test_predictions_list, best_index_list)
        best_test_predictions = test_predictions_list[best_index_list[0]]

    return combined_predictions, best_predictions,\
        combined_test_predictions, best_test_predictions


def get_next_best_single_fold(session, event, predictions_list, ground_truths,
                              best_index_list, min_improvement=0.0):
    """Find the next best model on a single fold.

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
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    predictions_list : list of :class:`rampwf.prediction_types.BasePrediction`
        Each element of the list is an instance of Predictions of a model
        on the same (cross-validation valid) data points.
    ground_truths : :class:`rampwf.prediction_types.BasePrediction`
        The ground truth.
    best_index_list : list of integers
        Indices of the current best model.
    min_improvement : float
        The mimimum improvement needed to continue the greedy loop.

    Returns
    -------
    best_index_list : list of integers
        Indices of the models in the new combination. If the same as input,
        no models wer found improving the score.
    """

    Predictions = type(predictions_list[0])
    score_type = event.get_official_score_type(session)
    score_function = score_type.score_function
    is_lower_the_better = score_type.is_lower_the_better

    best_predictions = Predictions.combine(predictions_list, best_index_list)
    best_score = score_function(ground_truths, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles
    r = np.arange(len(predictions_list))
    # Randomization doesn't matter, only in case of exact equality.
    # np.random.shuffle(r)
    # print r
    for i in r:
        index_list = np.append(best_index_list, i)
        combined_predictions = Predictions.combine(
            predictions_list, index_list)
        new_score = score_function(ground_truths, combined_predictions)
        if (is_lower_the_better and new_score < best_score) or\
                (not is_lower_the_better and new_score > best_score):
            best_predictions = combined_predictions
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index), best_score
    else:
        return best_index_list, best_score
