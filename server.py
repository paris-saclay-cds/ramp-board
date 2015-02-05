#!/usr/bin/env python2
from flask import Flask, request, redirect, url_for
from git import Repo, Submodule
import os.path

app = Flask(__name__)
repo = Repo('repo')

@app.route("/list/")
def list_submodules():
    if len(repo.submodules) == 0:
        return "No submodule found"
    else:
        html_list = "<ul>"
        for submodule in repo.submodules:
            html_list += "<li>{} - {}</li>".format(submodule.path, submodule.head)
        html_list += "</ul>"
        return html_list

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
    app.run(debug=True)
    print list_submodules()
