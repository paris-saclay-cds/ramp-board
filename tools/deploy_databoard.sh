#!/bin/bash
# :Usage: bash deploy_databoard.sh
# Prepare Ubuntu (14.04) server instance for the application deployment 
# There must be a file env.sh containing all required environment variables:
#    export DATABOARD_DB_NAME='databoard'
#    export DATABOARD_DB_USER='xxxx'
#    export DATABOARD_DB_PASSWORD='yyyy'
#    export DATABOARD_DB_URL='postgresql://xxxx:yyyy@localhost/databoard'
#    export SCIENCEFS_LOGIN='zzzz'
#    export DATARUN_URL='uuuu'
#    export DATARUN_USERNAME='vvvv'
#    export DATARUN_PASSWORD='wwww'
#    export NB_WORKERS=2


# Add environment variables
# env.sh file with environment variables must be in the same folder as this script
mv env.sh ~/.aliases
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.zshrc
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.zshrc
echo 'source ~/.aliases' >> ~/.bashrc
echo 'source ~/.aliases' >> ~/.zshrc
source ~/.bashrc

# Set databoard and persistent disk path
export DISK_PATH=/dev/vdb
export DATABOARD_PATH=/mnt/ramp_data/
#export LAST_DB_DUMP=databoard_XXXXX.dump

# Update Packages from the Ubuntu Repositories 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" update 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade 
# Install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py 
pip install pyopenssl ndg-httpsclient pyasn1
# Install Ubuntu dependencies for Python
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install build-essential python-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install gfortran
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install swig
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libatlas-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install liblapack-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libfreetype6 libfreetype6-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libxft-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install pandoc
# Install numpy, scipy, and ipython
pip install numpy
pip install scipy
pip install ipython
# Install Apache and mod_wsgi
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install apache2 libapache2-mod-wsgi
# Install git 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install git
# Install xgboost
cd; git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install
cd

# Mount ScienceFS disk
apt-get -y install sshfs
mkdir /mnt/datacamp
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp

# Format persistent disk (if first use)
#mkfs.ext4 $DIS_PATH
# Mount persistent disk
mkdir $DATABOARD_PATH
mount $DISK_PATH $DATABOARD_PATH

# Copy all databoard files (not the code, but submissions, ...)
cd $DATABOARD_PATH
mkdir datacamp
cp -r /mnt/datacamp/backup/databoard ${DATABOARD_PATH}/datacamp/.

# Clone the project
mkdir code
cd code
git clone https://camille24@bitbucket.org/kegl/databoard.git
cd databoard

# Install Postgres
# sudo apt-get -y install python-dev libpq-dev postgresql postgresql-contrib
sudo apt-get -y install libpq-dev postgresql postgresql-contrib
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
pip install -r requirements.txt
python setup.py develop

# Reacreate the database
python manage.py db upgrade
# For the first transfer from sqlite to postgres, use export_to_csv.sh and convert_to_postgres.py
# Then: 
# pg_restore -j 8 -U postgres -d databoard $LAST_DB_DUMP

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
sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_URL')#'$DATARUN_URL'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_USERNAME')#'$DATARUN_USERNAME'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_PASSWORD')#'$DATARUN_PASSWORD'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 

# Start celery workers
bash ${DATABOARD_PATH}/code/databoard/tools/celery_worker.sh start $NB_WORKERS

# Wrapping up some permissions issues
sudo chown -R :www-data ../.
sudo chown -R www-data:www-data ${DATABOARD_PATH}/datacamp/databoard


# Restart Apache
sudo service apache2 restart

# Install zsh
cd
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
echo 'source ~/.aliases' >> ~/.zshrc
sed -i 's/plugins=(git)/plugins=(git cp tmux screen pip lol fabric)/g' ~/.zshrc

# Add backup for the production server:
# execution of tools/dump_db.sh and tools/housekeeping.sh with crontab
# Not executed in the script because backup path must be checked and no backup for
# test server
# Add these lines to the file opened by crontab -e
# 02 0    * * *   root    bash /mnt/ramp_data/code/databoard/tools/dump_db.sh
# 22 1    * * *   root    bash /mnt/ramp_data/code/databoard/tools/housekeeping.sh

