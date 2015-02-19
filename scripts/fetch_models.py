# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import sys
import git
import glob
import uuid 
import shutil
import contextlib
import numpy as np
import pandas as pd

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from config_databoard import repos_path, root_path, tag_len_limit


sys.path.insert(1, os.path.join(prog_path, 'models'))

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


@contextlib.contextmanager  
def changedir(dir_name):
    current_dir = os.getcwd()
    try:
        os.chdir(dir_name)
        yield
    except Exception as e:
        print(e)  # to be replaced with proper logging
    finally:
        os.chdir(current_dir)


for rp in repo_paths:
    print(rp)

    if not os.path.isdir(rp):
        continue

    try:
        team_name = os.path.basename(rp)
        repo = git.Repo(rp)
        repo_path = os.path.join(submissions_path, team_name)
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)
        open(os.path.join(repo_path, '__init__.py'), 'a').close()

        if len(repo.tags) > 0:
            for t in repo.tags:
                tag_name = t.name

                # tag_name = tag_name.replace(',', ';')
                # tag_name = tag_name.replace(' ', '_')
                # tag_name = tag_name.replace('.', '->')
 
                unique_tag = str(uuid.uuid3(uuid.NAMESPACE_DNS, tag_name))
                tag_name_alias = 'm_{}'.format(unique_tag)

                model_path = os.path.join(repo_path, tag_name)
                copy_git_tree(t.object.tree, model_path)
                open(os.path.join(model_path, '__init__.py'), 'a').close()

                with changedir(repo_path):
                    if os.path.islink(tag_name_alias):
                        os.unlink(tag_name_alias)    
                    os.symlink(tag_name, tag_name_alias)

                relative_model_path = os.path.join(team_name, tag_name)
                relative_alias_path = os.path.join(team_name, tag_name_alias)
                tags_info.append([team_name, 
                                  tag_name, 
                                  t.commit.committed_date, 
                                  relative_alias_path,
                                  relative_model_path])
        else:
            print('No tag found for %s' % team_name)
    except Exception, e:
        print("Error: %s" % e)

if len(tags_info) > 0:
    columns = ['team', 'model', 'timestamp', 'path', 'alias']
    df = pd.DataFrame(np.array(tags_info), columns=columns)
    print(df)

    print('Writing submissions.csv file')
    df.to_csv('output/submissions.csv', index=False)
else:
    print('No submission found')
