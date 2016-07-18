#!/usr/bin/env bash
# Usage: bash export_to_csv.sh db_name

# obtains all data tables from database
TS=`sqlite3 $1 "SELECT tbl_name FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%';"`

# exports each table to csv
for T in $TS; do

  if [ "$T" = "cv_folds" ]; then 
sqlite3 $1 <<! 
.headers on
.mode csv
.output $T.csv
select id,type,hex(train_is),hex(test_is),event_id from $T;
!
  elif [ "$T" = "submission_on_cv_folds" ]; then 
sqlite3 $1 <<! 
.headers on
.mode csv
.output $T.csv
select id,submission_id,cv_fold_id,contributivity,best,hex(full_train_y_pred),hex(test_y_pred),train_time,valid_time,test_time,state,error_msg from $T;
!
  elif [ "$T" = "submission_scores" ]; then 
sqlite3 $1 <<! 
.headers on
.mode csv
.output $T.csv
select id,submission_id,event_score_type_id,valid_score_cv_bag,test_score_cv_bag,hex(valid_score_cv_bags),hex(test_score_cv_bags) from $T;
!
  else
sqlite3 $1 <<!
.headers on
.mode csv
.output $T.csv
select * from $T;
!
  fi
done
