import os
# import sys
# import shutil
import logging
import os.path
import datetime
import pandas as pd

from git import Repo
from zipfile import ZipFile
from collections import namedtuple
from flask import (
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
    flash,
    jsonify,
)

from databoard import app
# from databoard.model import columns, ModelState
from databoard.model import shelve_database
from databoard.generic import changedir
from databoard.config import repos_path
import databoard.config as config
import databoard.db.tools as db_tools

app.secret_key = os.urandom(24)
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output

logger = logging.getLogger('databoard')


def model_with_link(path_model):
    print path_model
    path, model, listing = path_model.split('+++++!*****')

    filename = listing.split('|')[0]
    filename_path = '/models/{}/{}'.format(path, filename)

    # if the tag name is too long, shrink it.
    if len(model) > tag_len_limit:
        model_trucated = model[:tag_len_limit] + '[...]'
        link = '<a href="{0}" class="popup" data-content="{1}">{2}</a>'.format(
            filename_path, model, model_trucated)
    else:
        link = '<a href="{0}">{1}</a>'.format(filename_path, model)
    return link


def error_local_to_url(path):
    filename = '/models/%s/error' % (path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


def timestamp_to_time(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


@app.route("/")
@app.route("/register")
def list_teams_repos():
    specific = config.config_object.specific

    RepoInfo = namedtuple('RepoInfo', 'name url')
    dir_list = filter(lambda x: not os.path.basename(x).startswith(
        '.'), os.listdir(repos_path))
    get_repo_url = lambda f: Repo(os.path.join(
        repos_path, f)).config_reader().get_value('remote "origin"', 'url')

    repo_list = []
    for f in dir_list:
        try:
            repo_list.append(RepoInfo(f, get_repo_url(f)))
        except Exception as e:
            logger.error(
                'Error when listing the repository: {}\n{}'.format(f, e))
    return render_template('list.html', submodules=repo_list, 
                           ramp_title=specific.ramp_title)


@app.route("/leaderboard")
def show_leaderboard():
    specific = config.config_object.specific

    leaderbord_html = db_tools.get_public_leaderboard()
    return render_template(
        'leaderboard.html', leaderboard=leaderbord_html,
        ramp_title=specific.ramp_title)


def show_leaderboard_old():
    specific = config.config_object.specific

    html_params = dict(escape=False,
                       index=False,
                       max_cols=None,
                       max_rows=None,
                       justify='left',
                       )
    table_classes = ['ui', 'blue', 'celled', 'table']
    sortable_table_classes = table_classes + ['sortable']

    # col_map = {'model': 'model <i class="help popup circle link icon" data-content="Click on the model name to view it"></i>'}
    common_columns = ['team', 'model', 'commit']
    # common_columns = ['team', col_map['model']]
    error_columns = common_columns + ['error']

    with shelve_database() as db:
        submissions_ = db['models']
        submissions_['commit'] = map(
            timestamp_to_time, submissions_['timestamp'])
        # 'inner' means intersection of the indices

        lb = submissions_.join(db['leaderboard1'], how='inner').\
            join(db['leaderboard2'], how='inner').\
            join(db['leaderboard_execution_times'], how='inner')
        failed = submissions_[submissions_.state == "error"]
        new_models = submissions_[submissions_.state == "new"]

    if len(submissions_) == 0:  # or len(l1) == 0 or len(l2) == 0:
        # flash('No models submitted yet.')
        return redirect(url_for('list_teams_repos'))

    failed.sort(columns='timestamp', inplace=True, ascending=True)
    new_models.sort(columns='timestamp', inplace=True, ascending=True)

    # l1.index = range(1, len(l1) + 1)
    # l2.index = range(1, len(l2) + 1)
    # failed.index = range(1, len(failed) + 1)
    # new_models.index = range(1, len(new_models) + 1)

    failed.loc[:, "error"] = failed.team + "/" + failed.index
    failed.loc[:, "error"] = failed.error.map(error_local_to_url)

    # adding the rank column
    lb.sort(columns='score', inplace=True, ascending=False)
    lb['rank'] = range(1, len(lb) + 1)

    for df in [lb, failed, new_models]:
        # dirty hack
        # create a new column 'path_model' and use to generate the link
        df['path_model'] = df.team + "/" + df.index + '+++++!*****' + \
            df.model + '+++++!*****' + df.listing
        df.model = df.path_model.map(model_with_link)
        # df.rename(
        #     columns=col_map,
        #     inplace=True)

    scores_columns = ['rank'] + common_columns + ['score']
    scores_columns += ['contributivity', 'train time', 'test time']
    lb_html = lb.to_html(
        columns=scores_columns, classes=sortable_table_classes, **html_params)
    new_html = new_models.to_html(
        columns=common_columns, classes=table_classes, **html_params)

    # if failed.shape[0] == 0:
    #     failed_html = None
    # else:
    failed_html = failed.to_html(
        columns=error_columns, classes=table_classes, **html_params)

    # if new_models.shape[0] == 0:
    #     new_html = None
    import databoard.db.tools as db_tools
    db_tools.print_users()
    db_tools.print_active_teams()
    db_tools.print_submissions()
    submissions = db_tools.get_submissions()
    for submission in submissions:
        submission.trained_state = 'scored'
    lb_html = db_tools.get_public_leaderboard()

    if '_' in request.path:
        return jsonify(leaderboard=lb_html,
                       failed_models=failed_html,
                       new_models=new_html)
    else:
        return render_template('leaderboard.html',
                               leaderboard=lb_html,
                               failed_models=failed_html,
                               new_models=new_html,
                               ramp_title=specific.ramp_title)

# FIXME Djalel: I just copy-pasted this a la Robi. Could be factorized with
# previous function I suppose
# TODO: should be accesible only by admins


@app.route("/_privateleaderboard")
@app.route("/privateleaderboard")
def show_private_leaderboard():
    specific = config.config_object.specific

    html_params = dict(escape=False,
                       index=False,
                       max_cols=None,
                       max_rows=None,
                       justify='left',
                       )
    table_classes = ['ui', 'blue', 'celled', 'table']
    sortable_table_classes = table_classes + ['sortable']

    # col_map = {'model': 'model <i class="help popup circle link icon" data-content="Click on the model name to view it"></i>'}
    common_columns = ['team', 'model', 'commit']
    # common_columns = ['team', col_map['model']]
    error_columns = common_columns + ['error']

    with shelve_database() as db:
        submissions = db['models']
        submissions['commit'] = map(
            timestamp_to_time, submissions['timestamp'])
        # 'inner' means intersection of the indices

        lb = submissions.join(db['leaderboard_classical_test'], how='inner').\
            join(db['leaderboard2'], how='inner').\
            join(db['leaderboard_execution_times'], how='inner')
        failed = submissions[submissions.state == "error"]
        new_models = submissions[submissions.state == "new"]

    if len(submissions) == 0:  # or len(l1) == 0 or len(l2) == 0:
        # flash('No models submitted yet.')
        return redirect(url_for('list_teams_repos'))

    failed.sort(columns='timestamp', inplace=True, ascending=True)
    new_models.sort(columns='timestamp', inplace=True, ascending=True)

    # l1.index = range(1, len(l1) + 1)
    # l2.index = range(1, len(l2) + 1)
    # failed.index = range(1, len(failed) + 1)
    # new_models.index = range(1, len(new_models) + 1)

    failed.loc[:, "error"] = failed.team + "/" + failed.index
    failed.loc[:, "error"] = failed.error.map(error_local_to_url)

    # adding the rank column
    lb.sort(columns='score', inplace=True, ascending=False)
    lb['rank'] = range(1, len(lb) + 1)

    for df in [lb, failed, new_models]:
        # dirty hack
        # create a new column 'path_model' and use to generate the link
        df['path_model'] = df.team + "/" + df.index + '+++++!*****' + \
            df.model + '+++++!*****' + df.listing
        df.model = df.path_model.map(model_with_link)
        # df.rename(
        #     columns=col_map,
        #     inplace=True)

    scores_columns = ['rank'] + common_columns + ['score']
    if 'calib score' in lb.columns:
        scores_columns += ['calib score']
    scores_columns += ['contributivity', "train time", "test time"]
    lb_html = lb.to_html(
        columns=scores_columns, classes=sortable_table_classes, **html_params)
    new_html = new_models.to_html(
        columns=common_columns, classes=table_classes, **html_params)

    # if failed.shape[0] == 0:
    #     failed_html = None
    # else:
    failed_html = failed.to_html(
        columns=error_columns, classes=table_classes, **html_params)

    # if new_models.shape[0] == 0:
    #     new_html = None

    if '_' in request.path:
        return jsonify(leaderboard=lb_html,
                       failed_models=failed_html,
                       new_models=new_html)
    else:
        return render_template('leaderboard.html',
                               leaderboard=lb_html,
                               failed_models=failed_html,
                               new_models=new_html,
                               ramp_title=specific.ramp_title)


@app.route('/submissions/<team_name>/<summission_hash>/<f_name>')
@app.route('/submissions/<team_name>/<summission_hash>/<f_name>/raw')
def view_model(team_name, summission_hash, f_name):
    """Rendering submission codes using templates/submission.html. The code of
    f_name is displayed in the left panel, the list of submissions files
    is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all
    the submission files and download them (managed here).


    Parameters
    ----------
    team_name : string
        The team.name of the submission.
    summission_hash : string
        The hash_ of the submission.
    f_name : string
        The name of the submission file

    Returns
    -------
    leaderboard : html string
        The rendered submission.html page.
    """
    from databoard.db.model import db, Team, Submission
    specific = config.config_object.specific

    team = db.session.query(Team).filter_by(name=team_name).one()
    submission = db.session.query(Submission).filter_by(
        team=team, hash_=summission_hash).one()
    submission_abspath = os.path.abspath(submission.path)
    archive_filename = 'archive.zip'

    if request.path.split('/')[-1] == 'raw':
        with changedir(submission_abspath):
            with ZipFile(archive_filename, 'w') as archive:
                for submission_file in submission.submission_files:
                    archive.write(submission_file.name)

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename='{}_{}_{}'.format(
                team_name, summission_hash[:6], f_name),
            mimetype='application/octet-stream')

    archive_url = '/submissions/{}/{}/{}/raw'.format(
        team_name, summission_hash, os.path.basename(archive_filename))

    submission_url = request.path.rstrip('/') + '/raw'
    submission_f_name = os.path.join(submission_abspath, f_name)
    if not os.path.exists(submission_f_name):
        return redirect(url_for('show_leaderboard'))

    with open(submission_f_name) as f:
        code = f.read()

    return render_template(
        'submission.html',
        code=code,
        submission_url=submission_url,
        submission_f_names=submission.submission_f_names,
        archive_url=archive_url,
        f_name=f_name,
        submission_name=submission.name,
        team_name=team.name,
        ramp_title=specific.ramp_title)


@app.route("/add/", methods=("POST",))
def add_team_repo():
    if request.method == "POST":
        repo_name = request.form["name"].strip()
        repo_path = request.form["url"].strip()
        message = ''

        correct_name = True
        for name_part in repo_name.split('_'):
            if not name_part.isalnum():
                correct_name = False
                message = 'Incorrect team name. Please only use letters, digits and underscores.'
                break

        if correct_name:
            try:
                Repo.clone_from(
                    repo_path, os.path.join(repos_path, repo_name))
            except Exception as e:
                logger.error('Unable to add a repository: \n{}'.format(e))
                message = str(e)

        if message:
            flash(message)

        return redirect(url_for('list_teams_repos'))
