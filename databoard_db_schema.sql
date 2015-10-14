-- Schema for databoard database.

create table users (
    user_id                  integer primary key autoincrement not null,
    user_name                text unique not null,
    linked_in_profile        text not null, -- we should make it mandatory
    access_level             integer default 0 -- admin = 1, ordinary_user = 0
    creation_timestamp       date default current_timestamp
);

-- We will also add individual members as single-user teams with their 
-- user name, when they sign up for a ramp. They can change their team name
-- for each ramp.
-- When users sign up, we check if they participate in ramps.max_teams_per_user
-- teams already, and propose them to delete a team from a ramp. We keep their 
-- submissions but set is_team_valid to 0 and don't show them on the leaderboard.
create table teams (
    team_id                  integer primary key autoincrement not null,
    team_name                text unique not null,
    is_team_valid            integer default 0, -- valid = 1, invalid = 0
    max_team_size            integer default 1,
    creation_timestamp       date default current_timestamp
);

create table team_members (
    team_id                  integer foreign key,
    user_id                  integer foreign key,
    access_level             integer default 0 -- team_admin = 1, ordinary_team_member = 0
    join_timestamp           date default current_timestamp
);

-- Should data sets be in the db or we should just store path? See the discussion
-- below, at the submissions table.
create table data_sets (
    data_set_id              integer primary key autoincrement not null,
    data_set_name            text unique not null,
    data_set_path            text not null
);

-- Like specifying targets, label list (for classification), train-test cut, etc.
-- Should provide read_data, prepare_data, get_train_data, get_test_data, 
-- get_check_data (see el nino) of specific.
create table data_setups (
    data_setup_id            integer primary key autoincrement not null,
    data_set_id              integer foreign key,
    data_setup_code          text not null -- python code implementing the setup
);

-- Should provide get_cv from specific. Should we actually store train/test
-- indices? See discussion at the submissions table.
create table cvs (
    cv_id                    integer primary key autoincrement not null,
    cv_name                  text unique not null,
    cv_code                  text not null -- python code implementing the cv
);

create table pipelines (
    pipeline_id              integer primary key autoincrement not null,
    pipeline_name            text unique not null
);

-- The actual analytics steps of the pipelines, like classifier, 
-- ts_feature_extractor, calibrator, etc.
-- Eventually I could imagine an OO hierarchy here (models that specialize)
-- but for now it is a flat table
create table elementary_steps (
    elementary_step_id       integer primary key autoincrement not null,
    elementary_step_name     text unique not null,
    elementary_step_train    text not null, -- python code implementing the training of the pipeline element
    elementary_step_test     text not null, -- python code implementing the testing of the pipeline element
    elementary_step_check    text not null, -- python code implementing the checking of the pipeline element (see el Nino)
    elementary_step_example  text -- optional python code implementing the pipeline element
);

-- The actual elementary models in a given pipeline
create table pipeline_elements (
    pipeline_id              integer foreign key,
    elementary_model_id      integer foreign key,
    order                    integer not null -- the order in which pipeline elements should be called
    is_mandatory             integer default 1 -- mandatory = 1, optional = 0 
);

-- Problem is a problem (challenge), ramp is an event (potentially extended in 
-- length). Ie., we can repeat the same problem is several ramps.
create table problems (
    problem_id               integer primary key autoincrement not null,
    problem_name             text unique not null,
    problem_description      text not null, -- short description for website
    data_setup_id            integer foreign key,
    cv_id                    integer foreign key,
    pipeline_id              integer foreign key,
    prediction_type_id       integer foreign key,
    creation_timestamp       date default current_timestamp
);

-- Security of train_server will have to be figured out. I mean: it is possible
-- that the ramp admin will not want us to be able to log in to those machines.
-- It is also possible that they don't want to disclose the data, just keep 
-- it on their train_server. It is OK, the data is needed only for train/test,
-- data_set_path is a path on the train_server, and we only need to get the 
-- prediction vectors at our side. Can be messy if we decide to store the data
-- and the predictions in the database (see discussion at the submissions table).
create table train_servers (
    train_server_id          integer primary key autoincrement not null,
    train_server_name        text unique not null,
    train_server_user_name   text unique not null,
    train_server_ssh_key     text unique not null, -- alternatively pwd?
    train_server_root        text unique not null
);

create table ramps (
    ramp_id                  integer primary key autoincrement not null,
    problem_id               integer foreign key,
    train_server_id          integer foreign key,
    ramp_name                text default null -- might want to overwrite problem_name
    ramp_description         text default null -- might want to overwrite problem_description
    max_teams_per_user       integer default 1,
    creation_timestamp       date default current_timestamp,
    opening_timestamp        date default null,
    public_opening_timestamp date default null, -- before that teams can see only their own scores
    closing_timestamp        date default null
);

-- eg. accuracy, logistic loss, RMSE, relative RMSE
create table score_types (
    score_type_id            integer primary key autoincrement not null,
    score_name               text unique not null,
    score_code               text not null -- python class implementing the score_type
);

-- eg. real scalar, probability table, etc.
create table prediction_types (
    prediction_type_id       integer primary key autoincrement not null,
    prediction_type_name     text unique not null,
    prediction_type_code     text not null -- python class implementing the prediction_type
);

-- when setting up a ramp, ramp_admin will first select prediction_type, then 
-- we propose one or more compatible score_types from this table
create table prediction_score_compatibility (
    prediction_type_id       integer foreign key,
    score_type_id            integer foreign key
);

-- For specifying which scores will appear in the leaderboard view
create table ramp_score_types (
    ramp_id                  integer foreign key,
    score_type_id             integer foreign key,
    ramp_score_name          text default null -- might want to overwrite score_name
);

create table ramp_participants (
    ramp_id                  integer foreign key,
    team_id                  integer foreign key,
    signup_timestamp         date default current_timestamp
);

create table ramp_admins (
    ramp_id                  integer foreign key,
    user_id                  integer foreign key,
    signup_timestamp         date default current_timestamp
);

create table models (
    model_id                 integer primary key autoincrement not null,
    elementary_step_id       integer foreign key, -- the model type like "classifier"
    team_id                  integer foreign key,
    model_code               text not null, -- python class implementing the elementary_step
    creation_timestamp       date default current_timestamp
);

-- Big question: should we put the predictions into the db? Lot of arrays. At 
-- this point I'd keep the current setup: the path where the train, valid, test
-- predictions are, are computed from the submission_id (since they are unique,
-- we can simply take the submission id). The clean db solution would be to
-- include the data and the cv folds into the db (instance by instance) so train
-- and test instances would get their ids, and predictions would correspond 
-- to the individual instances. We would have to add tables each time we have new
-- problem (data set) and a new ramp. It's complex, but we would have a much more
-- refined access to the predictions through the db.
create table submissions (
    submission_id            integer primary key autoincrement not null,
    team_id                  integer foreign key, -- we repeat it here for easy join
    submission_state         integer default 0, -- new = 0, trained = 1, tested = 2, error = 3, evaluated = 4, ignore = 5
    submission_timestamp     date default, current_timestamp
    training_timestamp       date default null
);

-- should check that user submits all mandatory elementary steps, and that the 
-- team_id of submission_id and model_id are the same
create table submission_elements (
    submission_id            integer foreign key,
    model_id                 integer foreign key
);

-- Leaderboards would be views of a join on submission_scores, submissions, and
-- teams. We can easily control access during the private period.
create table submission_scores (
    submission_id            integer foreign key,
    score_type_id            integer foreign key,
    score                    numeric,
    evaluation_timestamp     date default current_timestamp
);



