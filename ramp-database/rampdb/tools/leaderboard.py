import numpy as np
import pandas as pd

from ..model.event import Event
from ..model.event import EventTeam
from ..model.submission import Submission
from ..model.team import Team

from .team import get_event_team_by_name

from .submission import get_bagged_scores
from .submission import get_scores
from .submission import get_submission_max_ram
from .submission import get_time

pd.set_option('display.max_colwidth', -1)

# TODO: reorder columns
# TODO: select only the official score for the bagged score
# TODO: refactor the competition leaderboard


def _compute_score(session, submissions, leaderboard_type, event_name,
                   with_links=True):
    record_score = []
    event = session.query(Event).filter_by(name=event_name).one()
    map_score_precision = {score_type.name: score_type.precision
                           for score_type in event.score_types}
    for sub in submissions:
        # take only max n bag
        df_scores_bag = get_bagged_scores(session, sub.id)
        highest_level = df_scores_bag.index.get_level_values('n_bag').max()
        df_scores_bag = df_scores_bag.loc[(slice(None), highest_level), :]
        df_scores_bag.index = df_scores_bag.index.droplevel('n_bag')
        df_scores_bag = df_scores_bag.round(map_score_precision)

        if leaderboard_type == 'private':
            df_scores = get_scores(session, sub.id)
            df_scores = df_scores.round(map_score_precision)

            df_time = get_time(session, sub.id)
            df_time = df_time.stack().to_frame()
            df_time.index = df_time.index.set_names(['fold', 'step'])
            df_time = df_time.rename(columns={0: 'time'})

            df = pd.concat([df_scores, df_time], axis=1)
            df_mean = df.groupby('step').mean()
            df_std = df.groupby('step').std()

            # select only the validation and testing steps and rename them to
            # public and private
            map_renaming = {'valid': 'public', 'test': 'private'}
            df_mean = (df_mean.loc[list(map_renaming.keys())]
                            .rename(index=map_renaming)
                            .stack().to_frame().T)
            df_std = (df_std.loc[list(map_renaming.keys())]
                            .rename(index=map_renaming)
                            .stack().to_frame().T)
            df_scores_bag = (df_scores_bag.rename(index=map_renaming)
                                        .stack().to_frame().T)

            df = pd.concat([df_scores_bag, df_mean, df_std], axis=1,
                        keys=['bag', 'mean', 'std'])
        else:
            df_time = get_time(session, sub.id)
            df_time = df_time.stack().to_frame()
            df_time.index = df_time.index.set_names(['fold', 'step'])
            df_time = df_time.rename(columns={0: 'time'})
            df_time = df_time.groupby('step').mean()

            # select only the validation and testing steps and rename them to
            # public and private
            map_renaming = {'valid': 'public', 'test': 'private'}
            df_time = (df_time.loc[list(map_renaming.keys())]
                            .rename(index=map_renaming)
                            .stack().to_frame().T)
            map_renaming = {'valid': 'public'}
            df_scores_bag = (df_scores_bag.rename(index=map_renaming)
                                          .stack().to_frame().T)

            df = pd.concat([df_scores_bag, df_time], axis=1,
                        keys=['bag', 'mean', 'std'])

        df.columns = df.columns.set_names(['stat', 'set', 'score'])

        # change the multi-index into a stacked index
        df.columns = df.columns.map(lambda x: " ".join(x))

        df['team'] = sub.team.name
        df['submission'] = sub.name_with_link if with_links else sub.name
        df['contributivity'] = int(round(100 * sub.contributivity))
        df['historical contributivity'] = int(round(
            100 * sub.historical_contributivity))
        df['max RAM [MB]'] = get_submission_max_ram(session, sub.id)
        df['submitted at (UTC)'] = pd.Timestamp(sub.submission_timestamp)
        record_score.append(df)

    df = pd.concat(record_score, axis=0, ignore_index=True, sort=False)

    if leaderboard_type == 'public':
        df = df.sort_values(
            "bag public {}".format(event.official_score_name),
            ascending=event.get_official_score_type(session).is_lower_the_better
        )
    else:
        df = df.sort_values(
            "bag private {}".format(event.official_score_name),
            ascending=event.get_official_score_type(session).is_lower_the_better
        )
    return df


def _compute_competition(session, submissions, event_name):
    event = session.query(Event).filter_by(name=event_name).one()
    score_type = event.get_official_score_type(session)
    score_name = event.official_score_name

    # construct full leaderboard
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name[:20] for submission in submissions]
    leaderboard_df['public ' + score_name] = [
        round(
            submission.official_score.valid_score_cv_bag, score_type.precision)
        for submission in submissions]
    leaderboard_df['private ' + score_name] = [
        round(
            submission.official_score.test_score_cv_bag, score_type.precision)
        for submission in submissions]
    leaderboard_df['train time [s]'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['test time [s]'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        pd.Timestamp(submission.submission_timestamp)
        for submission in submissions]

    # select best submission for each team
    if score_type.is_lower_the_better:
        best_df = leaderboard_df.groupby('team').min()
    else:
        best_df = leaderboard_df.groupby('team').max()
    best_df = best_df[['public ' + score_name]].reset_index()
    best_df['best'] = True

    # merge to get a best indicator column then select best
    leaderboard_df = pd.merge(
        leaderboard_df, best_df, how='left',
        left_on=['team', 'public ' + score_name],
        right_on=['team', 'public ' + score_name])
    leaderboard_df = leaderboard_df.fillna(False)
    leaderboard_df = leaderboard_df[leaderboard_df['best']]
    leaderboard_df = leaderboard_df.drop(columns='best')

    # dealing with ties: we need the lowest timestamp
    best_df = leaderboard_df.groupby('team').min()
    best_df = best_df[['submitted at (UTC)']].reset_index()
    best_df['best'] = True
    leaderboard_df = pd.merge(
        leaderboard_df, best_df, how='left',
        left_on=['team', 'submitted at (UTC)'],
        right_on=['team', 'submitted at (UTC)'])
    leaderboard_df = leaderboard_df.fillna(False)
    leaderboard_df = leaderboard_df[leaderboard_df['best']]
    leaderboard_df = leaderboard_df.drop(columns='best')

    # sort by public score then by submission timestamp, compute rank
    leaderboard_df = leaderboard_df.sort_values(
        by=['public ' + score_name, 'submitted at (UTC)'],
        ascending=[score_type.is_lower_the_better, True])
    leaderboard_df['public rank'] = np.arange(len(leaderboard_df)) + 1

    # sort by private score then by submission timestamp, compute rank
    leaderboard_df = leaderboard_df.sort_values(
        by=['private ' + score_name, 'submitted at (UTC)'],
        ascending=[score_type.is_lower_the_better, True])
    leaderboard_df['private rank'] = np.arange(len(leaderboard_df)) + 1

    leaderboard_df['move'] =\
        leaderboard_df['public rank'] - leaderboard_df['private rank']
    leaderboard_df['move'] = [
        '{0:+d}'.format(m) if m != 0 else '-' for m in leaderboard_df['move']]

    public_leaderboard_df = leaderboard_df[[
        'public rank', 'team', 'submission', 'public ' + score_name,
        'train time [s]', 'test time [s]', 'submitted at (UTC)']]
    public_leaderboard_df = public_leaderboard_df.rename(columns={
        'public ' + score_name: score_name,
        'public rank': 'rank'
    })
    public_leaderboard_df = public_leaderboard_df.sort_values(by='rank')

    private_leaderboard_df = leaderboard_df[[
        'private rank', 'move', 'team', 'submission', 'private ' + score_name,
        'train time [s]', 'test time [s]', 'submitted at (UTC)']]
    private_leaderboard_df = private_leaderboard_df.rename(columns={
        'private ' + score_name: score_name,
        'private rank': 'rank'
    })
    private_leaderboard_df = private_leaderboard_df.sort_values(by='rank')

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    public_leaderboard_html = public_leaderboard_df.to_html(**html_params)
    private_leaderboard_html = private_leaderboard_df.to_html(**html_params)

    def table_format(xxx):
        return '<thead> {} </tbody>'.format(
            xxx.split('<thead>')[1].split('</tbody>')[0]
        )

    return (
        table_format(public_leaderboard_html),
        table_format(private_leaderboard_html)
    )


def get_leaderboard(session, leaderboard_type, event_name, user_name=None,
                    with_links=True):
    """Get a leaderboard.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    leaderboard_type : {'public', 'private', 'failed', 'new', 'competition'}
        The type of leaderboard to generate.
    event_name : str
        The event name.
    user_name : None or str, default is None
        The user name. If None, scores from all users will be queried.
    with_links : bool, default is True
        Whether or not the submission name should be clickable.

    Returns
    -------
    leaderboard : str
        The leaderboard in HTML format.
    """
    q = (session.query(Submission)
                .filter(Event.id == EventTeam.event_id)
                .filter(Team.id == EventTeam.team_id)
                .filter(EventTeam.id == Submission.event_team_id)
                .filter(Event.name == event_name))
    if user_name is not None:
        q = q.filter(Team.name == user_name)
    submissions = q.all()

    submission_filter = {'public': 'is_public_leaderboard',
                         'private': 'is_private_leaderboard',
                         'failed': 'is_error',
                         'new': 'is_new',
                         'competition': 'is_in_competition'}

    submissions = [sub for sub in submissions
                   if getattr(sub, submission_filter[leaderboard_type])]

    if not submissions:
        return None

    if leaderboard_type in ['public', 'private']:
        df = _compute_score(session, submissions, leaderboard_type, event_name,
                            with_links=with_links)
    elif leaderboard_type in ['new', 'failed']:
        columns = ['team',
                   'submission',
                   'submitted at (UTC)']

        if leaderboard_type == 'failed':
            columns.append('error')

        # we rely on the zip function ignore the submission state if the error
        # column was not appended
        data = [
            {column: value
            for column, value in zip(columns,
                                     [sub.event_team.team.name,
                                     sub.name_with_link,
                                     pd.Timestamp(sub.submission_timestamp),
                                     sub.state_with_link])}
            for sub in submissions
        ]
        df = pd.DataFrame(data, columns=columns)
    else:
        return _compute_competition(session, submissions, event_name)

    df_html = df.to_html(escape=False, index=False, max_cols=None,
                         max_rows=None, justify='left')
    df_html = '<thead> {} </tbody>'.format(
        df_html.split('<thead>')[1].split('</tbody>')[0]
    )
    return df_html


def update_leaderboards(session, event_name):
    """Update the leaderboards for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    """
    private_leaderboard_html = get_leaderboard(session, 'private', event_name)
    public_leaderboard_with_links = get_leaderboard(session, 'public',
                                                    event_name)
    public_leaderboard_without_links = get_leaderboard(session, 'public',
                                                       event_name,
                                                       with_links=False)
    failed_leaderboard_html = get_leaderboard(session, 'failed', event_name)
    new_leaderboard_html = get_leaderboard(session, 'new', event_name)
    competition_leaderboards_html = get_leaderboard(session, 'competition',
                                                    event_name)

    event = session.query(Event).filter_by(name=event_name).one()
    event.private_leaderboard_html = private_leaderboard_html
    event.public_leaderboard_html_with_links = public_leaderboard_with_links
    event.public_leaderboard_html_no_links = public_leaderboard_without_links
    event.failed_leaderboard_html = failed_leaderboard_html
    event.new_leaderboard_html = new_leaderboard_html
    event.public_competition_leaderboard_html = \
        competition_leaderboards_html[0]
    event.private_competition_leaderboard_html = \
        competition_leaderboards_html[1]
    session.commit()


def update_user_leaderboards(session, event_name, user_name):
    """Update the of a user leaderboards for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name. If None, scores from all users will be queried.
    """
    public_leaderboard_with_links = get_leaderboard(
        session, 'public', event_name, user_name
    )
    failed_leaderboard_html = get_leaderboard(
        session, 'failed', event_name, user_name
    )
    new_leaderboard_html = get_leaderboard(
        session, 'new', event_name, user_name
    )

    event_team = get_event_team_by_name(session, event_name, user_name)
    event_team.leaderboard_html = public_leaderboard_with_links
    event_team.failed_leaderboard_html = failed_leaderboard_html
    event_team.new_leaderboard_html = new_leaderboard_html
    session.commit()


def update_all_user_leaderboards(session, event_name):
    """Update the leaderboards for all users for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    """
    event = session.query(Event).filter_by(name=event_name).one()
    event_teams = session.query(EventTeam).filter_by(event=event).all()
    for event_team in event_teams:
        user_name = event_team.team.name

        public_leaderboard_with_links = get_leaderboard(
            session, 'public', event_name, user_name
        )
        failed_leaderboard_html = get_leaderboard(
            session, 'failed', event_name, user_name
        )
        new_leaderboard_html = get_leaderboard(
            session, 'new', event_name, user_name
        )

        event_team.leaderboard_html = public_leaderboard_with_links
        event_team.failed_leaderboard_html = failed_leaderboard_html
        event_team.new_leaderboard_html = new_leaderboard_html
    session.commit()
