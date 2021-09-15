import os
import logging
from pathlib import Path
import pandas as pd

from rampwf.utils import blend_submissions

from ..model import SubmissionSimilarity

from ._query import select_event_by_name
from ._query import select_submissions_by_state
from ._query import select_submission_by_id

logger = logging.getLogger("RAMP-DATABASE")


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
    submissions = select_submissions_by_state(session, event_name, state="scored")
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    for s in submissions:
        s.historical_contributivity = 0.0
    for s in submissions:
        s.historical_contributivity += s.contributivity
        similarities = (
            session.query(SubmissionSimilarity)
            .filter_by(type="target_credit", target_submission=s)
            .all()
        )
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
                    source_submission.historical_contributivity += partial_credit
                    s.historical_contributivity -= partial_credit
                    processed_submissions.append(source_submission)
    session.commit()


def compute_contributivity(
    session,
    event_name,
    ramp_kit_dir,
    ramp_data_dir,
    ramp_predictions_dir=None,
    min_improvement=0.0,
):
    """Blend submissions of an event, compute combined score and
       contributivities.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event associated to the submission.
    ramp_kit_dir : str
        The directory of the RAMP kit.
    ramp_data_dir : str
        The directory of the data.
    ramp_predictions_dir : str
        The directory with predictions
    min_improvement : float, default is 0.0
        The minimum improvement under which greedy blender is stopped.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Combining models")

    event = select_event_by_name(session, event_name)
    ramp_submission_dir = event.path_ramp_submissions
    score_type = event.get_official_score_type(session)

    submissions = select_submissions_by_state(session, event_name, state="scored")
    if len(submissions) == 0:
        logger.info("No submissions to blend.")
        return
    # ramp-board submission folder layout is different to that of
    # ramp-worklow. Here we symlink
    # submissions/submissions_<id>/training_output
    # to predictions/sumbmission_<id>/ if it exists, in order to avoid
    # rescoring the model.
    for sub in submissions:
        if ramp_predictions_dir is None or not Path(ramp_predictions_dir).exists():
            continue
        training_output_dir_board = Path(ramp_predictions_dir) / sub.basename
        training_output_dir_ramwf = (
            Path(ramp_submission_dir) / sub.basename / "training_output"
        )

        if (
            not training_output_dir_ramwf.exists()
            and training_output_dir_board.exists()
        ):
            # Note: on Windows 10+ this requires to enable the Developer Mode
            os.symlink(training_output_dir_board.resolve(), training_output_dir_ramwf)

    blend_submissions(
        submissions=[sub.basename for sub in submissions],
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        ramp_submission_dir=ramp_submission_dir,
        save_output=True,
        min_improvement=min_improvement,
    )

    bsc_f_name = "bagged_scores_combined.csv"
    bsc_df = pd.read_csv(
        os.path.join(ramp_submission_dir, "training_output", bsc_f_name)
    )
    n_folds = len(bsc_df) // 2

    row = (bsc_df["step"] == "valid") & (bsc_df["n_bag"] == n_folds - 1)
    event.combined_combined_valid_score = bsc_df[row][score_type.name].values[0]
    row = (bsc_df["step"] == "test") & (bsc_df["n_bag"] == n_folds - 1)
    event.combined_combined_test_score = bsc_df[row][score_type.name].values[0]

    bsfb_f_name = "bagged_scores_foldwise_best.csv"
    bsfb_df = pd.read_csv(
        os.path.join(ramp_submission_dir, "training_output", bsfb_f_name)
    )
    row = (bsfb_df["step"] == "valid") & (bsfb_df["n_bag"] == n_folds - 1)
    event.combined_foldwise_valid_score = bsfb_df[row][score_type.name].values[0]
    row = (bsfb_df["step"] == "test") & (bsfb_df["n_bag"] == n_folds - 1)
    event.combined_foldwise_test_score = bsfb_df[row][score_type.name].values[0]

    c_f_name = "contributivities.csv"
    contributivities_df = pd.read_csv(
        os.path.join(ramp_submission_dir, "training_output", c_f_name)
    )

    logger.info(contributivities_df)
    for index, row in contributivities_df.iterrows():
        sub_id = int(row["submission"][-9:])
        submission = select_submission_by_id(session, sub_id)
        submission.contributivity = 0.0
        for fold_i in range(n_folds):
            c_i = row["fold_{}".format(fold_i)]
            submission.contributivity += c_i

    session.commit()
