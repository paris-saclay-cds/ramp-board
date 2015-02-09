# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import glob
import os
import time
import git
import numpy as np
import pandas as pd 

from config_databoard import repos_path, root_path

base_path = repos_path
repo_paths = sorted(glob.glob(os.path.join(base_path, '*')))

submissions_path = os.path.join(root_path, 'models')

if not os.path.exists(submissions_path):
    os.mkdir(submissions_path)

tags_info = []

for rp in repo_paths:
    if not os.path.isdir(rp):
        continue

    team_name = os.path.basename(rp)
    repo = git.Repo(rp)
    o = repo.remotes.origin
    o.pull()
    print rp

    try:
        if len(repo.tags) > 0:
            for t in repo.tags:
                tag_name = t.name
                c = t.commit
                tree = repo.tree(c.hexsha)
                b = tree['model.py']
                file_content = b.data_stream.read()
                model_path = os.path.join(submissions_path, team_name)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_path = os.path.join(model_path, tag_name)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_path = os.path.join(model_path, 'model.py')
                with open(model_path, 'w') as f:
                    f.write(file_content)
                relative_model_path = os.path.join(team_name, tag_name)
                tags_info.append([team_name, tag_name, c.committed_date, relative_model_path])
        else:
            print('No tag found for %s' % team_name)
    except Exception, e:
        print "Error: %s" % e

if len(tags_info) > 0:
    columns = ['team', 'model', 'timestamp', 'path']
    df = pd.DataFrame(np.array(tags_info), columns=columns)
    print df

    print('Writing submission.csv file')
    df.to_csv('submissions.csv', index=False)
else:
    print('No submission found')

