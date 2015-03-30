import os
import sys
import shutil
import logging
import os.path
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
from .model import shelve_database, columns, ModelState
from .generic import changedir
from .config_databoard import (
    root_path, 
    repos_path, 
    serve_port,
    server_name,
    local_deployment,
    tag_len_limit,
    models_path,
)

app.secret_key = os.urandom(24)
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output

logger = logging.getLogger('databoard')

# TMP:
# app.config['SERVER_NAME'] = server_name + ':' + str(os.getenv('SERV_PORT', port))

def model_with_link(path_model):
    path, model = path_model.split()
    filename = '/models/%s/model.py' % (path)

    # if the tag name is too long, shrink it.
    if len(model) > tag_len_limit:
        model_trucated = model[:tag_len_limit] + '[...]'
        link = '<a href="{0}" class="popup" data-content="{1}">{2}</a>'.format(filename, model, model_trucated)
    else:
        link = '<a href="{0}">{1}</a>'.format(filename, model)
    return link


def error_local_to_url(path):
    filename = '/models/%s/error' % (path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


@app.route("/")
@app.route("/register")
def list_teams_repos():
    RepoInfo = namedtuple('RepoInfo', 'name url') 
    dir_list = filter(lambda x: not os.path.basename(x).startswith('.'), os.listdir(repos_path))
    get_repo_url = lambda f: Repo(os.path.join(repos_path, f)).config_reader().get_value('remote "origin"', 'url')

    repo_list = [] 
    for f in dir_list:
        try:
            repo_list.append(RepoInfo(f, get_repo_url(f)))
        except Exception as e:
            logger.error('Error when listing the repository: {}\n{}'.format(f, e))
    return render_template('list.html', submodules=repo_list)

@app.route("/_leaderboard")
@app.route("/leaderboard")
def show_leaderboard():
    html_params = dict(escape=False,
                       index=True,
                       max_cols=None,
                       max_rows=None,
                       justify='left',
                       classes=['ui', 'table', 'blue'],
                       )

    with shelve_database() as db:
        submissions = db['models']

        l1 = submissions.join(db['leaderboard1'], how='inner')  # 'inner' means intersection of the indices
        l2 = submissions.join(db['leaderboard2'], how='inner')
        failed = submissions[submissions.state == "error"]
        new_models = submissions[submissions.state == "new"]

        if len(submissions) == 0: # or len(l1) == 0 or len(l2) == 0:
            # flash('No models submitted yet.')
            return redirect(url_for('list_teams_repos'))

    l1.index = range(1, len(l1) + 1)
    l2.index = range(1, len(l2) + 1)
    failed.index = range(1, len(failed) + 1)
    new_models.sort(columns='timestamp', inplace=True, ascending=True)
    new_models.index = range(1, len(new_models) + 1)

    failed.loc[:, "error"] = failed.path
    failed.loc[:, "error"] = failed.error.map(error_local_to_url)

    # col_map = {'model': 'model <i class="help popup circle link icon" data-content="Click on the model name to view it"></i>'}

    for df in [l1, l2, failed, new_models]:
        # dirty hack
        # create a new column 'path_model' and use to generate the link
        df['path_model'] = df.path + ' ' + df.model
        df.model = df.path_model.map(model_with_link)
        # df.rename(
        #     columns=col_map, 
        #     inplace=True)


    common_columns = ['team', 'model']
    # common_columns = ['team', col_map['model']]
    scores_columns = common_columns + ['score']
    error_columns = common_columns + ['error']
    l1_html = l1.to_html(columns=scores_columns, **html_params)
    l2_html = l2.to_html(columns=scores_columns, **html_params)
    new_html = new_models.to_html(columns=common_columns, **html_params)

    # if failed.shape[0] == 0:
    #     failed_html = None
    # else:
    failed_html = failed.to_html(columns=error_columns, **html_params)

    # if new_models.shape[0] == 0:
    #     new_html = None

    if '_' in request.path:
        return jsonify(leaderboard_1=l1_html,
                       leaderboard_2=l2_html,
                       failed_models=failed_html,
                       new_models=new_html)
    else:
        return render_template('leaderboard.html', 
                               leaderboard_1=l1_html,
                               leaderboard_2=l2_html,
                               failed_models=failed_html,
                               new_models=new_html)


@app.route('/models/<team>/<tag>/<filename>')
@app.route('/models/<team>/<tag>/<filename>/raw')
def view_model(team, tag, filename):
    directory = os.path.join(models_path, team, tag)
    directory = os.path.abspath(directory)
    archive_filename = 'archive.zip'
    archive_url = '/models/{}/{}/{}/raw'.format(team, tag, os.path.basename(archive_filename))
   
    if filename == 'error':
        filename += '.txt'
    if filename == archive_filename:
        with shelve_database() as db:
            listing = db['models'].loc[tag, 'listing'].split('|')
        with changedir(directory):
            with ZipFile(archive_filename, 'w') as archive:
                for f in listing:
                    archive.write(f)

    
    if request.path.split('/')[-1] == 'raw':
        return send_from_directory(directory,
                                   filename, 
                                   as_attachment=True,
                                   attachment_filename='{}_{}_{}'.format(team, tag[:6], filename), 
                                   mimetype='application/octet-stream')

    model_url = request.path.rstrip('/') + '/raw'
    model_file = os.path.join(directory, filename)
    if not os.path.exists(model_file):
        return redirect(url_for('show_leaderboard'))
    
    with open(model_file) as f:
        code = f.read()
    with shelve_database() as db:
        models = db['models']
        listing = models.loc[tag, 'listing'].split('|')
        model_name = models.loc[tag, 'model']
   
    return render_template(
        'model.html', 
        code=code, 
        model_url=model_url,
        listing=listing,
        archive_url=archive_url,
        filename=filename,
        model_name=model_name,
        team_name=team)

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
                Repo.clone_from(repo_path, os.path.join(repos_path, repo_name))
            except Exception as e:
                logger.error('Unable to add a repository: \n{}'.format(e))
                message = str(e)

        if message:
            flash(message)

        return redirect(url_for('list_teams_repos'))
