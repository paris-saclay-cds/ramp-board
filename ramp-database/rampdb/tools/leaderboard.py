import pandas as pd

from ..model.event import Event
from ..model.event import EventTeam
from ..model.submission import Submission
from ..model.team import Team

from .submission import get_bagged_scores
from .submission import get_scores
from .submission import get_submission_max_ram
from .submission import get_time


def get_private_leaderboards(session, event_name, user_name=None):
    """Create private leaderboards.

    Parameters
    ----------
    ession : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name.

    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """
    q = (session.query(Submission)
                .filter(Event.id == EventTeam.event_id)
                .filter(Team.id == EventTeam.team_id)
                .filter(EventTeam.id == Submission.event_team_id)
                .filter(Event.name == event_name))
    if user_name is not None:
        q = q.filter(Team.name == user_name)
    submissions = q.all()
    submissions = [submission for submission in submissions
                   if submission.is_private_leaderboard]
    event = Event.query.filter_by(name=event_name).one()

    df_scores = get_scores(session, )

    score_names = [score_type.name for score_type in event.score_types]
    scoresss = np.array([
        [[round(score.valid_score_cv_bag, score.precision),
          round(score.valid_score_cv_mean, score.precision),
          round(score.valid_score_cv_std, score.precision + 1),
          round(score.test_score_cv_bag, score.precision),
          round(score.test_score_cv_mean, score.precision),
          round(score.test_score_cv_std, score.precision + 1)]
         for score in submission.ordered_scores(score_names)]
        for submission in submissions
    ])
    if len(submissions) > 0:
        scoresss = np.swapaxes(scoresss, 0, 1)
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name_with_link for submission in submissions]
    for score_name in score_names:  # to make sure the column is created
        leaderboard_df[score_name + ' pub bag'] = 0
        leaderboard_df[score_name + ' pub mean'] = 0
        leaderboard_df[score_name + ' pub std'] = 0
        leaderboard_df[score_name + ' pr bag'] = 0
        leaderboard_df[score_name + ' pr mean'] = 0
        leaderboard_df[score_name + ' pr std'] = 0
    for score_name, scoress in zip(score_names, scoresss):
        leaderboard_df[score_name + ' pub bag'] = scoress[:, 0]
        leaderboard_df[score_name + ' pub mean'] = scoress[:, 1]
        leaderboard_df[score_name + ' pub std'] = scoress[:, 2]
        leaderboard_df[score_name + ' pr bag'] = scoress[:, 3]
        leaderboard_df[score_name + ' pr mean'] = scoress[:, 4]
        leaderboard_df[score_name + ' pr std'] = scoress[:, 5]
    leaderboard_df['contributivity'] = [
        int(round(100 * submission.contributivity))
        for submission in submissions]
    leaderboard_df['historical contributivity'] = [
        int(round(100 * submission.historical_contributivity))
        for submission in submissions]
    leaderboard_df['train time [s]'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['trt std'] = [
        int(round(submission.train_time_cv_std))
        for submission in submissions]
    leaderboard_df['test time [s]'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['tet std'] = [
        int(round(submission.valid_time_cv_std))
        for submission in submissions]
    leaderboard_df['max RAM [MB]'] = [
        int(round(submission.max_ram)) if type(submission.max_ram) == float
        else 0
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]
    sort_column = event.official_score_name + ' pr bag'
    leaderboard_df = leaderboard_df.sort_values(
        sort_column, ascending=event.official_score_type.is_lower_the_better)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)

    # logger.info(u'private leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

return table_format(leaderboard_html)