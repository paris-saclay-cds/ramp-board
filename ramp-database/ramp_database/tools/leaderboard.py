from distutils.version import LooseVersion
from itertools import product

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

width = -1 if LooseVersion(pd.__version__) < LooseVersion("1.0.0") else None
pd.set_option('display.max_colwidth', width)


def _compute_leaderboard(session, submissions, leaderboard_type, event_name,
                         with_links=True):
    """Format the leaderboard.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submissions : list of :class:`ramp_database.model.Submission`
        The submission to report in the leaderboard.
    leaderboard_type : {'public', 'private'}
        The type of leaderboard to built.
    event_name : str
        The name of the event.
    with_links : bool
        Whether or not the submission name should be clickable.

    Returns
    -------
    leaderboard : dataframe
        The leaderboard in a dataframe format.
    """
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

        df_scores = get_scores(session, sub.id)
        df_scores = df_scores.round(map_score_precision)

        df_time = get_time(session, sub.id)
        df_time = df_time.stack().to_frame()
        df_time.index = df_time.index.set_names(['fold', 'step'])
        df_time = df_time.rename(columns={0: 'time'})
        df_time = df_time.sum(axis=0, level="step").T

        df_scores_mean = df_scores.groupby('step').mean()
        df_scores_std = df_scores.groupby('step').std()

        # select only the validation and testing steps and rename them to
        # public and private
        map_renaming = {'valid': 'public', 'test': 'private'}
        df_scores_mean = (df_scores_mean.loc[list(map_renaming.keys())]
                                        .rename(index=map_renaming)
                                        .stack().to_frame().T)
        df_scores_std = (df_scores_std.loc[list(map_renaming.keys())]
                                      .rename(index=map_renaming)
                                      .stack().to_frame().T)
        df_scores_bag = (df_scores_bag.rename(index=map_renaming)
                                      .stack().to_frame().T)

        df = pd.concat([df_scores_bag, df_scores_mean, df_scores_std], axis=1,
                       keys=['bag', 'mean', 'std'])

        df.columns = df.columns.set_names(['stat', 'set', 'score'])

        # change the multi-index into a stacked index
        df.columns = df.columns.map(lambda x: " ".join(x))

        # add the aggregated time information
        df_time.index = df.index
        df_time = df_time.rename(
            columns={'train': 'train time [s]',
                     'valid': 'validation time [s]',
                     'test': 'test time [s]'}
        )
        df = pd.concat([df, df_time], axis=1)

        if leaderboard_type == 'private':
            df['submission ID'] = sub.basename.replace('submission_', '')
        df['team'] = sub.team.name
        df['submission'] = sub.name_with_link if with_links else sub.name
        df['contributivity'] = int(round(100 * sub.contributivity))
        df['historical contributivity'] = int(round(
            100 * sub.historical_contributivity))
        df['max RAM [MB]'] = get_submission_max_ram(session, sub.id)
        df['submitted at (UTC)'] = pd.Timestamp(sub.submission_timestamp)
        record_score.append(df)

    # stack all the records
    df = pd.concat(record_score, axis=0, ignore_index=True, sort=False)

    # keep only second precision for the time stamp
    df['submitted at (UTC)'] = df['submitted at (UTC)'].astype('datetime64[s]')

    # reordered the column
    stats_order = (['bag', 'mean', 'std'] if leaderboard_type == 'private'
                   else ['bag'])
    dataset_order = (['public', 'private'] if leaderboard_type == 'private'
                     else ['public'])
    score_order = ([event.official_score_name] +
                   [score_type.name for score_type in event.score_types
                    if score_type.name != event.official_score_name])
    score_list = [
        '{} {} {}'.format(stat, dataset, score)
        for dataset, score, stat in product(dataset_order,
                                            score_order,
                                            stats_order)
    ]
    # Only display train and validation time for the public leaderboard
    time_list = (['train time [s]', 'validation time [s]', 'test time [s]']
                 if leaderboard_type == 'private'
                 else ['train time [s]', 'validation time [s]'])
    col_ordered = (
        ['team', 'submission'] +
        score_list +
        ['contributivity', 'historical contributivity'] +
        time_list +
        ['max RAM [MB]', 'submitted at (UTC)']
    )
    if leaderboard_type == "private":
        col_ordered = ["submission ID"] + col_ordered
    df = df[col_ordered]

    # check if the contributivity columns are null
    contrib_columns = ['contributivity', 'historical contributivity']
    if (df[contrib_columns] == 0).all(axis=0).all():
        df = df.drop(columns=contrib_columns)

    df = df.sort_values(
        "bag {} {}".format(leaderboard_type, event.official_score_name),
        ascending=event.get_official_score_type(session).is_lower_the_better
    )

    # rename the column name for the public leaderboard
    if leaderboard_type == 'public':
        df = df.rename(columns={
            key: value for key, value in zip(score_list, score_order)
        })
    return df


def _compute_competition_leaderboard(session, submissions, leaderboard_type,
                                     event_name):
    """Format the competition leaderboard.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submissions : list of :class:`ramp_database.model.Submission`
        The submission to report in the leaderboard.
    leaderboard_type : {'public', 'private'}
        The type of leaderboard to built.
    event_name : str
        The name of the event.

    Returns
    -------
    competition_leaderboard : dataframe
        The competition leaderboard in a dataframe format.
    """
    event = session.query(Event).filter_by(name=event_name).one()
    score_type = event.get_official_score_type(session)
    score_name = event.official_score_name

    private_leaderboard = _compute_leaderboard(session, submissions, 'private',
                                               event_name, with_links=False)

    time_list = (['train time [s]', 'validation time [s]', 'test time [s]']
                 if leaderboard_type == 'private'
                 else ['train time [s]', 'validation time [s]'])

    col_selected_private = (['team', 'submission'] +
                            ['bag private ' + score_name,
                             'bag public ' + score_name] +
                            time_list +
                            ['submitted at (UTC)'])
    leaderboard_df = private_leaderboard[col_selected_private]
    leaderboard_df = leaderboard_df.rename(
        columns={'bag private ' + score_name: 'private ' + score_name,
                 'bag public ' + score_name: 'public ' + score_name}
    )

    # select best submission for each team
    best_df = (leaderboard_df.groupby('team').min()
               if score_type.is_lower_the_better
               else leaderboard_df.groupby('team').max())
    best_df = best_df[['public ' + score_name]].reset_index()
    best_df['best'] = True

    # merge to get a best indicator column then select best
    leaderboard_df = pd.merge(
        leaderboard_df, best_df, how='left',
        left_on=['team', 'public ' + score_name],
        right_on=['team', 'public ' + score_name]
    )
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

    leaderboard_df['move'] = \
        leaderboard_df['public rank'] - leaderboard_df['private rank']
    leaderboard_df['move'] = [
        '{:+d}'.format(m) if m != 0 else '-' for m in leaderboard_df['move']]

    col_selected = (
        [leaderboard_type + ' rank', 'team', 'submission',
         leaderboard_type + ' ' + score_name] +
        time_list +
        ['submitted at (UTC)']
    )
    if leaderboard_type == 'private':
        col_selected.insert(1, 'move')

    df = leaderboard_df[col_selected]
    df = df.rename(columns={
        leaderboard_type + ' ' + score_name: score_name,
        leaderboard_type + ' rank': 'rank'
    })
    df = df.sort_values(by='rank')
    return df


def get_leaderboard(session, leaderboard_type, event_name, user_name=None,
                    with_links=True):
    """Get a leaderboard.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    leaderboard_type : {'public', 'private', 'failed', 'new', \
'public competition', 'private competition'}
        The type of leaderboard to generate.
    event_name : str
        The event name.
    user_name : None or str, default is None
        The user name. If None, scores from all users will be queried. This
        parameter is discarded when requesting the competition leaderboard.
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
                         'public competition': 'is_in_competition',
                         'private competition': 'is_in_competition'}

    submissions = [sub for sub in submissions
                   if (getattr(sub, submission_filter[leaderboard_type]) and
                       sub.is_not_sandbox)]

    if not submissions:
        return None

    if leaderboard_type in ['public', 'private']:
        df = _compute_leaderboard(
            session, submissions, leaderboard_type, event_name,
            with_links=with_links
        )
    elif leaderboard_type in ['new', 'failed']:
        if leaderboard_type == 'new':
            columns = ['team', 'submission', 'submitted at (UTC)', 'state']
        else:
            columns = ['team', 'submission', 'submitted at (UTC)', 'error']

        # we rely on the zip function ignore the submission state if the error
        # column was not appended
        data = [{
            column: value for column, value in zip(
                columns,
                [sub.event_team.team.name,
                 sub.name_with_link,
                 pd.Timestamp(sub.submission_timestamp),
                 (sub.state_with_link if leaderboard_type == 'failed'
                  else sub.state)])
            } for sub in submissions]
        df = pd.DataFrame(data, columns=columns)
    else:
        # make some extra filtering
        submissions = [sub for sub in submissions if sub.is_public_leaderboard]
        if not submissions:
            return None
        competition_type = ('public' if 'public' in leaderboard_type
                            else 'private')
        df = _compute_competition_leaderboard(
            session, submissions, competition_type, event_name
        )

    df_html = df.to_html(escape=False, index=False, max_cols=None,
                         max_rows=None, justify='left')
    df_html = '<thead> {} </tbody>'.format(
        df_html.split('<thead>')[1].split('</tbody>')[0]
    )
    return df_html


def update_leaderboards(session, event_name, new_only=False):
    """Update the leaderboards for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    new_only : bool, default is False
        Whether or not to update the whole leaderboards or only the new
        submissions. You can turn this option to True when adding a new
        submission in the database.
    """
    event = session.query(Event).filter_by(name=event_name).one()
    if not new_only:
        event.private_leaderboard_html = get_leaderboard(
            session, 'private', event_name
        )
        event.public_leaderboard_html_with_links = get_leaderboard(
            session, 'public', event_name
        )
        event.public_leaderboard_html_no_links = get_leaderboard(
            session, 'public', event_name, with_links=False
        )
        event.failed_leaderboard_html = get_leaderboard(
            session, 'failed', event_name
        )
        event.public_competition_leaderboard_html = get_leaderboard(
            session, 'public competition', event_name
        )
        event.private_competition_leaderboard_html = get_leaderboard(
            session, 'private competition', event_name
        )
    event.new_leaderboard_html = get_leaderboard(
        session, 'new', event_name
    )
    session.commit()


def update_user_leaderboards(session, event_name, user_name,
                             new_only=False):
    """Update the of a user leaderboards for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name. If None, scores from all users will be queried.
    new_only : bool, default is False
        Whether or not to update the whole leaderboards or only the new
        submissions. You can turn this option to True when adding a new
        submission in the database.
    """
    event_team = get_event_team_by_name(session, event_name, user_name)
    if not new_only:
        event_team.leaderboard_html = get_leaderboard(
            session, 'public', event_name, user_name
        )
        event_team.failed_leaderboard_html = get_leaderboard(
            session, 'failed', event_name, user_name
        )
    event_team.new_leaderboard_html = get_leaderboard(
        session, 'new', event_name, user_name
    )
    session.commit()


def update_all_user_leaderboards(session, event_name, new_only=False):
    """Update the leaderboards for all users for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    new_only : bool, default is False
        Whether or not to update the whole leaderboards or only the new
        submissions. You can turn this option to True when adding a new
        submission in the database.
    """
    event = session.query(Event).filter_by(name=event_name).one()
    event_teams = session.query(EventTeam).filter_by(event=event).all()
    for event_team in event_teams:
        user_name = event_team.team.name
        if not new_only:
            event_team.leaderboard_html = get_leaderboard(
                session, 'public', event_name, user_name
            )
            event_team.failed_leaderboard_html = get_leaderboard(
                session, 'failed', event_name, user_name
            )
        event_team.new_leaderboard_html = get_leaderboard(
            session, 'new', event_name, user_name
        )
    session.commit()
