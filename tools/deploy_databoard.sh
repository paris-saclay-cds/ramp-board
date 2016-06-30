#!/bin/bash
# :Usage: bash deploy_databoard.sh
# Prepare Ubuntu (14.04) server instance for the application deployment 
# We follow steps (+ other steps) from 
# - https://www.digitalocean.com/community/tutorials/how-to-serve-django-applications-with-apache-and-mod_wsgi-on-ubuntu-14-04

# Add environment variables
# env.sh file with environment variables must be in the same folder as this script
mv env.sh ~/.aliases
echo 'source ~/.aliases' >> ~/.bashrc
echo 'source ~/.aliases' >> ~/.zshrc
source ~/.bashrc

# Set locales variables
export LC_ALL=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

# Set databoard and persistent disk path
export DISK_PATH=/dev/vdb
export DATABOARD_PATH=/mnt/ramp_data/

# Install Packages from the Ubuntu Repositories 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" update 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install python-pip apache2 libapache2-mod-wsgi
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install git
# wget https://raw.github.com/brainsik/virtualenv-burrito/master/virtualenv-burrito.sh
# bash virtualenv-burrito.sh 
# source /root/.venvburrito/startup.sh
# Install Numpy, Scipy, and xgboost
sudo apt-get -y install python-numpy python-scipy  
sudo apt-get -y install python-setuptools
cd; git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install
cd

# Mount ScienceFS disk
apt-get -y install sshfs
mkdir /mnt/datacamp
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp

# Mount persistent disk
mkdir $DATABOARD_PATH
mount $DISK_PATH $DATABOARD_PATH

# Copy all databoard files (not the code, but submissions, ...)
cd $DATABOARD_PATH
mkdir datacamp
# cp -r /mnt/datacamp/backup/databoard ${DATABOARD_PATH}/datacamp/.
cp -r /mnt/datacamp/databoard ${DATABOARD_PATH}/datacamp/.

# Clone the project
mkdir code
cd code
git clone -b postgres2 --single-branch https://camille24@bitbucket.org/kegl/databoard.git
cd databoard

# Install Postgres
sudo apt-get -y install python-dev libpq-dev postgresql postgresql-contrib
pg_createcluster 9.3 main --start
# Change postgres permissions
sed -i "85c local   all             postgres                                trust" /etc/postgresql/9.3/main/pg_hba.conf 
sudo service postgresql restart
psql -U postgres -c '\i tools/setup_database.sql'
# Pb with db user password..not properly set with the above script... workaround:
psql -U postgres -c "ALTER ROLE $DATABOARD_DB_USER WITH PASSWORD '$DATABOARD_DB_PASSWORD'"
# Change database user permissions
sed -i "86i local   all             $DATABOARD_DB_USER                                 trust" /etc/postgresql/9.3/main/pg_hba.conf
sudo service postgresql restart

# Install required Python packages
pip install -Ur requirements.txt
python setup.py develop

# Reacreate the database
python -c 'from databoard import db; db.create_all()'
# TODO add script here to recreate existing db 
# Depends on how we ll make db backup. 
# For the first transfer from sqlite to postgres, use export_to_csv.sh and convert_to_postgres.py

# Configure Apache: copy apache conf file to /etc/apache2/sites-available/
mv /etc/apache2/sites-available/000-default.conf /etc/apache2/sites-available/backup.conf
cp tools/databoard.conf /etc/apache2/sites-available/000-default.conf

# Set the ServerName
export IP_MASTER=$(/sbin/ifconfig eth0 | grep "inet addr" | awk -F: '{print $2}' | awk '{print $1}')
sed -i "s/<ServerName>/$IP_MASTER/g" /etc/apache2/sites-available/000-default.conf

# Deal with environment variables
# In databoard, we need to set up one one environment variable: 
# DATABOARD_DB_URL, which is called in databoard/config.py
# The usual way is to define environment variable in the apache conf file:
sed -i "3a SetEnv DATABOARD_DB_URL $DATABOARD_DB_URL" /etc/apache2/sites-available/000-default.conf;
sed -i "s/SetEnv/    SetEnv/g" /etc/apache2/sites-available/000-default.conf
# But for some reasons, it does not work. 
# So we set up the value of this environment variable directly in databoard/config.py 
sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" /home/code/databoard/databoard/config.py 

# Wrapping up some permissions issues
sudo chown -R :www-data ../.
sudo chown -R www-data:www-data /home/datacamp/databoard


# Restart Apache
sudo service apache2 restart


