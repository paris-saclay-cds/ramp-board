import os
import sys
import logging
import os.path
import pandas as pd

from git import Repo
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
from .config_databoard import (
    root_path, 
    repos_path, 
    serve_port,
    server_name,
    local_deployment,
    tag_len_limit,
)

app.secret_key = os.urandom(24)
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output

logger = logging.getLogger('databoard')

def model_local_to_url(path):
    filename = '%s/models/%s' % (server_name, path)
    link = '<a href="{0}">Code</a>'.format(filename)
    return link

def model_with_link(path_model):
    path, model = path_model.split()
    filename = '%s/models/%s' % (server_name, path)

    # if the tag name is too long, shrink it.
    if len(model) > tag_len_limit:
        model_trucated = model[:tag_len_limit] + '[...]'
        link = '<a href="{0}" class="popup" data-content="{1}">{2}</a>'.format(filename, model, model_trucated)
    else:
        link = '<a href="{0}">{1}</a>'.format(filename, model)
    return link


def error_local_to_url(path):
    filename = '%s/models/%s/error' % (server_name, path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


@app.route("/")
@app.route("/register")
def list_teams_repos():
    RepoInfo = namedtuple('RepoInfo', 'name url') 
    dir_list = filter(lambda x: not x.startswith('.'), os.listdir(repos_path))
    get_repo_url = lambda f: Repo(os.path.join(repos_path, f)).config_reader().get_value('remote "origin"', 'url')
    dir_list = [RepoInfo(f, get_repo_url(f)) for f in dir_list]
    return render_template('list.html', submodules=dir_list)

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

    # if not all((os.path.exists("output/leaderboard1.csv"),
    #             os.path.exists("output/leaderboard2.csv"),
    #     )):
    #     return redirect(url_for('list_teams_repos'))

    with shelve_database() as db:
        submissions = db['models']
        l1 = submissions.join(db['leaderboard1'], how='outer')
        l2 = submissions.join(db['leaderboard2'], how='outer')
        failed = submissions[submissions.state == "error"]

    # l1 = pd.read_csv("output/leaderboard1.csv")
    # l2 = pd.read_csv("output/leaderboard2.csv")
    # failed = pd.read_csv("output/failed_submissions.csv")

    l1.index = range(1, len(l1) + 1)
    l2.index = range(1, len(l2) + 1)
    failed.index = range(1, len(failed) + 1)

    failed["error"] = failed.path
    failed["error"] = failed.error.map(error_local_to_url)

    col_map = {'model': 'model <i class="help popup circle link icon" data-content="Click on the model name to view it"></i>'}

    for df in [l1, l2, failed]:
        print df
        df['path_model'] = df.path + ' ' + df.model  # dirty hack
        df.model = df.path_model.map(model_with_link)
        df.rename(
            columns=col_map, 
            inplace=True)

    common_columns = ['team', col_map['model']]
    scores_columns = common_columns + ['score']
    error_columns = common_columns + ['error']
    html1 = l1.to_html(columns=scores_columns, **html_params)
    html2 = l2.to_html(columns=scores_columns, **html_params)

    if failed.shape[0] == 0:
        failed_html = None
    else:
        failed_html = failed.to_html(columns=error_columns, **html_params)

    if '_' in request.path:
        return jsonify(leaderboard_1=html1,
                       leaderboard_2=html2,
                       failed_models=failed_html)
    else:
        return render_template('leaderboard.html', 
                               leaderboard_1=html1,
                               leaderboard_2=html2,
                               failed_models=failed_html)


@app.route('/models/<team>/<tag>')
@app.route('/models/<team>/<tag>/raw')
def download_model(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    model_url = os.path.join(directory, 'model.py')

    if request.path.split('/')[-1] == 'raw':
        return send_from_directory(directory,
                                   'model.py', 
                                   as_attachment=True,
                                   attachment_filename='{}_{}.py'.format(team, tag),
                                   mimetype='application/octet-stream')
    else:
        with open(model_url) as f:
            code = f.read()
        return render_template('model.html', code=code, model_url=request.path.rstrip('/') + '/raw')


@app.route('/models/<team>/<tag>/error')
def download_error(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    error_url = os.path.join(directory, 'error.txt')

    if not os.path.exists(error_url):
        return redirect(url_for('show_leaderboard'))
    with open(error_url) as f:
        code = f.read()
    return render_template('model.html', code=code)

    directory = os.path.join(root_path, "models", team, tag)
    return send_from_directory(directory,
                               'error.txt',
                               mimetype='text/plain')


@app.route("/add/", methods=("POST",))
def add_team_repo():
    if request.method == "POST":
        repo_name = request.form["name"].strip()
        repo_path = request.form["url"].strip()
        message = ''
        try:
            git.Repo.clone_from(repo_path, repo_name)
        except Exception as e:
            message = str(e)

        if message:
            flash(message)
        return redirect(url_for('list_teams_repos'))
