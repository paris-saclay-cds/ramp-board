#!/usr/bin/env python2

import os
import sys
import os.path
import pandas as pd

from git import Repo, Submodule, BadName

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, prog_path)

from config_databoard import (
    root_path, 
    repos_path, 
    serve_port,
    server_name,
    local_deployment,
)
from flask import (
    Flask, 
    request, 
    redirect, 
    url_for, 
    render_template, 
    send_from_directory,
    flash,
    jsonify,
)


pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output

app = Flask(__name__)
repo = Repo(repos_path)
app.secret_key = os.urandom(24)


def model_local_to_url(path):
    filename = '%s/models/%s' % (server_name, path)
    link = '<a href="{0}">Code</a>'.format(filename)
    return link

def model_with_link(path_model):
    path, model = path_model.split()
    filename = '%s/models/%s' % (server_name, path)
    link = '<a href="{}">{}</a>'.format(filename, model)
    return link


def error_local_to_url(path):
    filename = '%s/models/%s/error' % (server_name, path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


@app.route("/")
@app.route("/register")
def list_submodules():
    try:
        sm = repo.submodules
    except BadName:
        sm = []
    return render_template('list.html', submodules=sm)

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

    if not all((os.path.exists("output/leaderboard1.csv"),
                os.path.exists("output/leaderboard2.csv"),
        )):
        return redirect(url_for('list_submodules'))

    l1 = pd.read_csv("output/leaderboard1.csv")
    l2 = pd.read_csv("output/leaderboard2.csv")
    failed = pd.read_csv("output/failed_submissions.csv")

    l1.index = range(1, len(l1) + 1)
    l2.index = range(1, len(l2) + 1)
    failed.index = range(1, len(failed) + 1)

    failed["error"] = failed.path
    failed["error"] = failed.error.map(error_local_to_url)

    col_map = {'model': 'model<i class="help circle link icon" data-content="Click on the model name to view it"></i>'}

    for df in [l1, l2, failed]:
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
def add_submodule():
    if request.method == "POST":
        submodule_name = request.form["name"].strip()
        submodule_path = request.form["url"].strip()
        message = ''
        try:
            Submodule.add(
                    repo = repo,
                    name = submodule_name,
                    path = submodule_name,
                    url = submodule_path,
                )
            repo.index.commit('Submodule added: {}'.format(submodule_name))
        except Exception as e:
            message = str(e)

        if message:
            flash(message)
        return redirect(url_for('list_submodules'))


if __name__ == "__main__":
    debug_mode = os.environ.get('DEBUGLB', local_deployment)
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
    app.run(debug=bool(debug_mode), port=serve_port, host='0.0.0.0')
