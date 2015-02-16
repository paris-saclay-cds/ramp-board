# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import git
import glob
import shutil
import numpy as np
import pandas as pd
from config_databoard import repos_path, root_path

base_path = repos_path
repo_paths = sorted(glob.glob(os.path.join(base_path, '*')))

submissions_path = os.path.join(root_path, 'models')

if not os.path.exists(submissions_path):
    os.mkdir(submissions_path)

tags_info = []


def copy_git_tree(tree, dest_folder):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    for file_elem in tree.blobs:
        with open(os.path.join(dest_folder, file_elem.name), 'w') as f:
            shutil.copyfileobj(file_elem.data_stream, f)
    for tree_elem in tree.trees:
        copy_git_tree(tree_elem, os.path.join(dest_folder, tree_elem.name))

        
for rp in repo_paths:
    print rp

    if not os.path.isdir(rp):
        continue

    try:
        team_name = os.path.basename(rp)
        repo = git.Repo(rp)
        o = repo.remotes.origin
        
        repo_path = os.path.join(submissions_path, team_name)
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)
        open(os.path.join(repo_path, '__init__.py'), 'a').close()

        if len(repo.tags) > 0:
            for t in repo.tags:
                tag_name = t.name

                tag_name = tag_name.replace(',', ';')
                tag_name = tag_name.replace(' ', '_')
                tag_name = tag_name.replace('.', '->')
 
                model_path = os.path.join(repo_path, tag_name)
                copy_git_tree(t.object.tree, model_path)
                open(os.path.join(model_path, '__init__.py'), 'a').close()
                
                relative_model_path = os.path.join(team_name, tag_name)
                tags_info.append([team_name, tag_name, t.commit.committed_date, relative_model_path])
        else:
            print('No tag found for %s' % team_name)
    except Exception, e:
        print "Error: %s" % e

if len(tags_info) > 0:
    columns = ['team', 'model', 'timestamp', 'path']
    df = pd.DataFrame(np.array(tags_info), columns=columns)
    print df

    print('Writing submissions.csv file')
    df.to_csv('output/submissions.csv', index=False)
else:
    print('No submission found')
