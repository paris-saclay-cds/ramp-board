# 29/06/16
# Script to add in the table Event official_score_name after backuping the
# old sqlite database
from databoard import db
from databoard.model import Event

score_name = {'drug_spectra': 'combined',
              'air_passengers_dssp4': 'rmse',
              'boston_housing_test': 'rmse',
              'epidemium2_cancer_mortality': 'rmse',
              'iris_test': 'acc',
              'HEP_detector_anomalies': 'auc'}

for kk, vv in score_name.items():
    event = Event.query.filter_by(name=kk).one()
    event.official_score_name = vv
    db.session.add(event)
    db.session.commit()
