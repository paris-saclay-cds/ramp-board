## Before running the script, create a backup of databoard/model.py
## (mv databoard/model.py databoard/model_backup.py)
## and move tools/model_transfer.py to databoard/model.py
import inspect
import math
import pandas as pd
from databoard import db
import databoard.model

dir_data = '/home/camille/Documents/Profiling/stratus_May2016/databoard/db/'
csv1 = ['submission_file_types.csv', 'users.csv', 'workflows.csv',
        'extensions.csv', 'score_types.csv']
csv2 = ['workflow_element_types.csv', 'submission_file_type_extensions.csv',
        'teams.csv', 'problems.csv']
csv3 = ['workflow_elements.csv', 'events.csv']
csv4 = ['cv_folds.csv', 'event_admins.csv', 'event_teams.csv',
        'event_score_types.csv']
csv5 = ['submissions.csv']
csv6 = ['submission_on_cv_folds.csv', 'submission_files.csv',
        'submission_similaritys.csv']
csv7 = ['user_interactions.csv', 'submission_score_on_cv_folds.csv']
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
    # table.__table__.columns.keys()
    data = pd.read_csv(csv_file)
    for dd in data.iterrows():
        table_data = dd[1].to_dict()
        # Deal with boolean: convert 0/1 to '0'/'1'
        for kk, vv in table_data.items():
            try:
                if table.__table__.columns[kk].type.python_type == bool:
                    table_data[kk] = str(vv)
            except NotImplementedError:
                pass
            if type(vv) == float:
                if math.isnan(vv):
                    table_data[kk] = None
        t = None
        try:
            t = table(**table_data)
#         except TypeError as e:
#             mess = "__init__() got an unexpected keyword argument '%s'"
#             if str(e) == mess % 'id':
#                 t, removed_field = clean_data(table, table_data, [])
#             elif (str(e) == mess % 'workflow_id' and
#                   table.__tablename__ == 'problems'):
#                 t, removed_field = clean_data(table, table_data,
#                                               ['workflow_id'])
#             elif str(e) == mess % 'admin_id':
#                 table_data = put_obj_not_id(table_data, 'admin',
#                                             databoard.model.User)
#                 t, removed_field = clean_data(table, table_data,
#                                               ['initiator_id', 'acceptor_id'])
#             elif (str(e) == mess % 'public_opening_timestamp' and
#                   table.__tablename__ == 'events'):
#                 add_event(table_data['name'])
#             elif (str(e) == mess % 'workflow_id' and
#                   table.__tablename__ == 'workflow_elements'):
#                 table_data = put_obj_not_id(table_data, 'workflow',
#                                             databoard.model.Workflow)
#                 table_data = put_obj_not_id(table_data, 'workflow_element_type',
#                                             databoard.model.WorkflowElementType)
#                 table_data['name_in_workflow'] = table_data['name']
#                 del table_data['name']
#                 t, removed_field = clean_data(table, table_data, [])
#             elif (str(e) == mess % 'event_id' and
#                   table.__tablename__ == 'event_teams'):
#                 table_data = put_obj_not_id(table_data, 'event',
#                                             databoard.model.Event)
#                 table_data = put_obj_not_id(table_data, 'team',
#                                             databoard.model.Team)
#                 t, removed_field = clean_data(table, table_data,
#                                               ['last_submission_name'])
#                 t.last_submission_name = removed_field['last_submission_name']
#             elif (str(e) == mess % 'test_time_cv_mean' and
#                   table.__tablename__ == 'submissions'):
#                 table_data = put_obj_not_id(table_data, 'event_team',
#                                             databoard.model.EventTeam)
#                 list_remove = table_data.keys()
#                 list_remove.remove('name')
#                 list_remove.remove('id')
#                 list_remove.remove('event_team')
#                 with db.session.no_autoflush:
#                     t, removed_field = clean_data(table, table_data,
#                                                   list_remove)
#                     for kk, vv in removed_field.items():
#                         if type(vv) == float:
#                             if math.isnan(vv):
#                                 vv = None
#                         setattr(t, kk, vv)
#             else:
#                 raise e
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
            csv_file = dir_data + dict_csv[name]
            create_table_instances(obj, csv_file, db)


def list_csv_to_dict(list_csv):
    dict_csv = {}
    for file_csv in list_csv:
        # convert csv file name to model name
        name_csv = ''.join([dd.title()
                            for dd in file_csv.split('.')[0].split('_')])
        name_csv = name_csv[:-1]  # remove the s at the end
        dict_csv[name_csv] = file_csv
    return dict_csv


for list_csv in meta_list_csv:
    print('*' * 80)
    print(list_csv)
    print('-' * 80)
    dict_csv = list_csv_to_dict(list_csv)
    create_tables_from_list_csv(databoard.model, dict_csv, dir_data, db)
