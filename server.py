#!/usr/bin/env python2

import os
import os.path
import pandas as pd

from git import Repo, Submodule
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
)


pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output

app = Flask(__name__)
repo = Repo(repos_path)


def model_local_to_url(path):
    filename = '%s/models/%s' % (server_name, path)
    link = '<a href="{0}">Code</a>'.format(filename)
    return link


def error_local_to_url(path):
    filename = '%s/models/%s/error' % (server_name, path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


@app.route("/")
@app.route("/register/")
def list_submodules():
    return render_template('list.html', submodules=repo.submodules)


@app.route("/leaderboard/")
def show_leaderboard():
    html_params = dict(escape=False,
                       index=True,
                       max_cols=None,
                       max_rows=None,
                       justify='left',
                       classes=['ui', 'table', 'red'])

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

    for df in [l1, l2, failed]:
        df.drop('timestamp', axis=1, inplace=True)
        df.path = df.path.map(model_local_to_url)

    html1 = l1.to_html(**html_params)
    html2 = l2.to_html(**html_params)

    if failed.shape[0] == 0:
        failed_html = None
    else:
        failed_html = failed.to_html(**html_params)

    return render_template('leaderboard.html', 
                           leaderboard_1=html1,
                           leaderboard_2=html2,
                           failed_models=failed_html)


@app.route('/models/<path:team>/<path:tag>')
def download_model(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    return send_from_directory(directory,
                               'model.py',
                               mimetype='text/x-script.python')


@app.route('/models/<path:team>/<path:tag>/error')
def download_error(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    return send_from_directory(directory,
                               'error.txt',
                               mimetype='text/x-script.python')


@app.route("/add/", methods=("POST",))
def add_submodule():
    if request.method == "POST":
        Submodule.add(
                repo = repo,
                name = request.form["name"],
                path = request.form["name"],
                url = request.form["url"],
            )
        return redirect(url_for('list_submodules'))


if __name__ == "__main__":
    debug_mode = os.environ.get('DEBUGLB', local_deployment)
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
    app.run(debug=bool(debug_mode), port=serve_port, host='0.0.0.0')
