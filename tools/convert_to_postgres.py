# Before running the script, create a backup of databoard/model.py
# (mv databoard/model.py databoard/model_backup.py)
# and move tools/model_transfer.py to databoard/model.py
# The csv files mentionned at the beginning are obtained with the script
# tools/export_to_csv.sh.
# If a csv file is too big (as it might be the case for
# submission_on_cv_folds.csv), you need to split it in different file, such as
# submission_on_cv_fold<i>.csv with i between 0 and 9
# (to split files, you can use split_csv.sh)
import inspect
import base64
import zlib
import math
import numpy as np
import pandas as pd
from databoard import db
import databoard.model

dir_data = './'
csv1 = [['submission_file_types.csv'], ['users.csv'], ['workflows.csv'],
        ['extensions.csv'], ['score_types.csv']]
csv2 = [['workflow_element_types.csv'], ['submission_file_type_extensions.csv'],
        ['teams.csv'], ['problems.csv']]
csv3 = [['workflow_elements.csv'], ['events.csv']]
csv4 = [['cv_folds.csv'], ['event_admins.csv'], ['event_teams.csv'],
        ['event_score_types.csv']]
csv5 = [['submissions.csv']]
csv6 = [['submission_on_cv_fold%s.csv' % i for i in range(1, 6)],
        ['submission_files.csv'],
        ['submission_similaritys.csv'], ['submission_scores.csv']]
csv7 = [['user_interactions.csv'], ['submission_score_on_cv_folds.csv']]
meta_list_csv = [csv1, csv2, csv3, csv4, csv5, csv6, csv7]


def clean_data(table, table_data, remove_field):
    t_id = table_data['id']
    del table_data['id']
    removed_field = {}
    for rf in remove_field:
        removed_field[rf] = table_data[rf]
        del table_data[rf]
    t = table(**table_data)
    t.id = t_id
    return t, removed_field


def put_obj_not_id(table_data, db_obj_name, model):
    db_obj_id = table_data['%s_id' % db_obj_name]
    del table_data['%s_id' % db_obj_name]
    db_obj = model.query.filter_by(id=db_obj_id).one()
    table_data[db_obj_name] = db_obj
    return table_data


def create_table_instances(table, csv_file, db):
    data = pd.read_csv(csv_file)
    for dd in data.iterrows():
        table_data = dd[1].to_dict()
        # Change hex() in keys that contain it
        if 'hex(train_is)' in table_data.keys():
            table_data['train_is'] = table_data.pop('hex(train_is)')
            table_data['test_is'] = table_data.pop('hex(test_is)')
        elif 'hex(full_train_y_pred)' in table_data.keys():
            table_data['full_train_y_pred'] =\
                table_data.pop('hex(full_train_y_pred)')
            table_data['test_y_pred'] = table_data.pop('hex(test_y_pred)')
        elif 'hex(valid_score_cv_bags)' in table_data.keys():
            table_data['valid_score_cv_bags'] =\
                table_data.pop('hex(valid_score_cv_bags)')
            table_data['test_score_cv_bags'] =\
                table_data.pop('hex(test_score_cv_bags)')
        # Deal with boolean: convert 0/1 to '0'/'1'
        # Deal with NumpyType()
        for kk, vv in table_data.items():
            try:
                if table.__table__.columns[kk].type.python_type == bool:
                    table_data[kk] = str(vv)
            except NotImplementedError:
                # NumpyType
                vv = base64.b16decode(vv)
                table_data[kk] = np.loads(zlib.decompress(vv))
            if type(vv) == float:
                if math.isnan(vv):
                    table_data[kk] = None
        t = None
        try:
            t = table(**table_data)
        except Exception as e:
            raise e
        if t:
            db.session.add(t)
        db.session.commit()
    table_name = table.__table__.name
    sql_c = ("SELECT setval('%s_id_seq', " +
             "(SELECT MAX(id) FROM %s) +1);") % (table_name, table_name)
    db.engine.execute(sql_c)


def create_tables_from_list_csv(model_module, dict_csv, dir_data, db):
    for name, obj in inspect.getmembers(model_module):
        if name in dict_csv.keys():
            print('Creating table %s' % name)
            for dd in dict_csv[name]:
                csv_file = dir_data + dd
                create_table_instances(obj, csv_file, db)


def list_csv_to_dict(list_csv):
    dict_csv = {}
    for list_file_csv in list_csv:
        file_csv = list_file_csv[0]
        # convert csv file name to model name
        name_csv = ''.join([dd.title()
                            for dd in file_csv.split('.')[0].split('_')])
        name_csv = name_csv[:-1]  # remove the s at the end
        # Deal with CVFold...
        if name_csv == 'CvFold':
            name_csv = 'CVFold'
        elif name_csv == 'SubmissionOnCvFold':
            name_csv = 'SubmissionOnCVFold'
        dict_csv[name_csv] = list_file_csv
    return dict_csv


for list_csv in meta_list_csv:
    print('*' * 80)
    print(list_csv)
    print('-' * 80)
    dict_csv = list_csv_to_dict(list_csv)
    create_tables_from_list_csv(databoard.model, dict_csv, dir_data, db)
