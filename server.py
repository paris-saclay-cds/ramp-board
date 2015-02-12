#!/usr/bin/env python2

import os.path
import pandas as pd
from git import Repo, Submodule
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from config_databoard import root_path, repos_path

app = Flask(__name__)
repo = Repo(repos_path)

serve_port = 8080
gt_path = os.path.join(root_path, 'ground_truth')

# to_html tends to truncate the output
pd.set_option('display.max_colwidth', -1)


def model_local_to_url(path):
    filename = '%s/models/%s' % (server_name, path)
    link = '<a href="{0}">Show the code</a>'.format(filename)
    return link


def error_local_to_url(path):
    filename = '%s/models/%s/error' % (server_name, path)
    link = '<a href="{0}">Show the error</a>'.format(filename)
    return link


@app.route("/")
@app.route("/register/")
@app.route("/list/")
def list_submodules():
    if len(repo.submodules) == 0:
        return render_template('list.html', submodules=repo.submodules)
    else:
        return render_template('list.html', submodules=repo.submodules)


@app.route("/leaderboard/")
def show_leaderboard_1():
    html_params = dict(escape=False,
                       index=False,
                       max_cols=None,
                       max_rows=None,
                       justify='left',
                       classes=['ui', 'blue', 'table'])

    l1 = pd.read_csv("leaderboard1.csv")
    l2 = pd.read_csv("leaderboard2.csv")
    failed = pd.read_csv("failed_submissions.csv")

    failed["error"] = failed.path
    failed["error"] = failed.error.map(error_local_to_url)

    for df in [l1, l2, failed]:
        df.drop('timestamp', axis=1, inplace=True)
        df.path = df.path.map(model_local_to_url)

    html1 = l1.to_html(**html_params)
    html2 = l2.to_html(**html_params)
    failed_html = failed.to_html(**html_params)

    return render_template('leaderboard.html', leaderboard_1=html1,
                           leaderboard_2=html2,
                           failed_models=failed_html)


@app.route('/models/<path:team>/<path:tag>')
def download_model(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    return send_from_directory(directory,
                               'model.py',
                               mimetype='text/x-script.phyton')


@app.route('/models/<path:team>/<path:tag>/error')
def download_error(team, tag):
    directory = os.path.join(root_path, "models", team, tag)
    return send_from_directory(directory,
                               'error.txt',
                               mimetype='text/x-script.phyton')


@app.route("/add/", methods=["GET", "POST"])
def add_submodule():
    if request.method == "POST":
        Submodule.add(
                repo = repo,
                name = request.form["name"],
                path = request.form["name"],
                url = request.form["url"],
            )
        return redirect(url_for('list_submodules'))
    else:
        sub_form = """
                    <form method="post">
                        <label>name</label>
                        <input type="text" name="name"/>
                        <label>git repository</label>
                        <input type="text" name="url"/>
                        <input type="submit" value="Add"/>
                    </form>
                   """
        return sub_form


if __name__ == "__main__":
    server_name = 'http://localhost:{}'.format(serve_port)
    app.run(debug=True, port=8080)

    # server_name = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(serve_port)
    # app.run(debug=False, port=serve_port, host='0.0.0.0')
    # app.run(debug=True, port=8080, host='127.0.0.1')
