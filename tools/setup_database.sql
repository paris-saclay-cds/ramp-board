\set database_name `echo "$DATABOARD_DB_NAME"`
\set database_user `echo "$DATABOARD_DB_USER"`
\set database_pwd `echo "$DATABOARD_DB_PASSWORD"`
CREATE DATABASE :database_name;
CREATE USER :database_user WITH PASSWORD ':database_pwd';
ALTER ROLE :database_user SET client_encoding TO 'utf8';
ALTER ROLE :database_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE :database_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE :database_name TO :database_user;

