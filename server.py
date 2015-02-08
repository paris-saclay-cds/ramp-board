#!/usr/bin/env python2

import pandas as pd
import os.path
from git import Repo, Submodule
from flask import Flask, request, redirect, url_for, render_template
from Submission.config import root_path, repos_path
from Submission.generic import leaderboard_classical, leaderboard_combination, leaderboard_to_html

app = Flask(__name__)

# team_repos = os.path.join(root_path, 'TeamsRepos')
team_repos = repos_path

repo = Repo(team_repos)

@app.route("/register/")
@app.route("/list/")
def list_submodules():
    if len(repo.submodules) == 0:
        return render_template('list.html', submodules=repo.submodules)
        # return "No submodule found"
    else:
        html_list = "<ul>"
        return render_template('list.html', submodules=repo.submodules)

@app.route("/leaderboad/1")
def show_leaderboard_1():
    submissions_path = os.path.join(root_path, 'Submission', 'trained_submissions.csv')
    trained_models = pd.read_csv(submissions_path)
    l1 = leaderboard_classical(trained_models)

    print leaderboard_to_html(l1)
    return leaderboard_to_html(l1)

@app.route("/leaderboad/2")
def show_leaderboard_2():
    submissions_path = os.path.join(root_path, 'Submission', 'trained_submissions.csv')
    trained_models = pd.read_csv(submissions_path)

    gt_path = os.path.join(root_path, 'Submission', 'GroundTruth')

    l2 = leaderboard_combination(trained_models, gt_path)
    return leaderboard_to_html(l2)

@app.route("/leaderboad/")
def show_leaderboard():
    submissions_path = os.path.join(root_path, 'Submission', 'trained_submissions.csv')
    trained_models = pd.read_csv(submissions_path)
    gt_path = os.path.join(root_path, 'Submission', 'GroundTruth')

    l1 = leaderboard_classical(trained_models)
    l2 = leaderboard_combination(trained_models, gt_path)

    return render_template('leaderboard.html', leaderboard_1=l1, leaderboard_2=l2)
    
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
    # app.run(debug=True, port=8080)
    app.run(debug=False, port=8080, host='0.0.0.0')
