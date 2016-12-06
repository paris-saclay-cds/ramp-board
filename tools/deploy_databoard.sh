#!/bin/bash
# Prepare Ubuntu (14.04) server instance for the application deployment 
#
# :Usage: bash deploy_databoard.sh {disk_path} {db_dump}
# {disk_path} : path to attached persistent disk
# {db_dump} : database dump from which to create new database . Give only the dump
#file name, this file should be located on the sciencefs disk in ~/databoard/backup 
#which will be mounted on the VM in /mnt/datacamp/backup
#
# There must be a file env.sh containing all required environment variables:
#    export DATABOARD_PATH='/mnt/ramp_data/'  #where to mount the persistent disk
#    export DATABOARD_DB_NAME='databoard'
#    export DATABOARD_DB_USER='xxxx'
#    export DATABOARD_DB_PASSWORD='yyyy'
#    export DATABOARD_DB_URL='postgresql://xxxx:yyyy@localhost/databoard'
#    export SCIENCEFS_LOGIN='zzzz'
#    export DATARUN_URL='uuuu'
#    export DATARUN_USERNAME='vvvv'
#    export DATARUN_PASSWORD='wwww'

DISK_PATH=$1
LAST_DB_DUMP=$2

# Add environment variables
# env.sh file with environment variables must be in the same folder as this script
mv env.sh ~/.aliases
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.zshrc
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.zshrc
echo 'source ~/.aliases' >> ~/.bashrc
echo 'source ~/.aliases' >> ~/.zshrc
export LC_ALL=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
source ~/.bashrc
source ~/.aliases

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
# Install python-netcdf4 (requires zlib, hdf5, and netCDF-C)
sudo apt-get -y install m4
wget http://zlib.net/zlib-1.2.8.tar.gz
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar
wget https://github.com/Unidata/netcdf-c/archive/v4.4.1.tar.gz
tar -xzvf zlib-1.2.8.tar.gz
tar -xvf hdf5-1.8.17.tar
tar -xzvf v4.4.1.tar.gz
cd zlib-1.2.8
export ZDIR=/usr/local
./configure --prefix=${ZDIR}
sudo make check
sudo make install 
cd ../hdf5-1.8.17
export H5DIR=/usr/local
./configure --with-zlib=${ZDIR} --prefix=${H5DIR}
sudo make check   # Fails here, but seems ok for netcdf
sudo make install
cd ../netcdf-c-4.4.1
export NCDIR=/usr/local
sudo CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR}
sudo make check
sudo make install  # or sudo make install
cd
sudo USE_SETUPCFG=0 pip install netcdf

# Mount ScienceFS disk
apt-get -y install sshfs
mkdir /mnt/datacamp
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp

# Format persistent disk (if first use)
#mkfs.ext4 $DISK_PATH
# Mount persistent disk
mkdir $DATABOARD_PATH
mount $DISK_PATH $DATABOARD_PATH

# Copy all databoard files (not the code, but submissions, ...)
cd $DATABOARD_PATH
mkdir datacamp
echo "Copying submission directory"
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

# Recreate the database
#python manage.py db upgrade
# For the first transfer from sqlite to postgres, use export_to_csv.sh and convert_to_postgres.py
# Then: 
pg_restore -j 8 -U postgres -d databoard /mnt/datacamp/backup/$LAST_DB_DUMP

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
cp ${DATABOARD_PATH}/code/databoard/databoard/config_local.py ${DATABOARD_PATH}/code/databoard/databoard/config.py
sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATABOARD_PATH', './')#'$DATABOARD_PATH'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_URL')#'$DATARUN_URL'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_USERNAME')#'$DATARUN_USERNAME'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
sed -i "s#os.environ.get('DATARUN_PASSWORD')#'$DATARUN_PASSWORD'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 
# Change server name in config
sed -i "s#current_server_name = '0.0.0.0:8080'#current_server_name = '$IP_MASTER'#g" ${DATABOARD_PATH}/code/databoard/databoard/config.py 

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

# Start celery workers
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install rabbitmq-server
mkdir ${DATABOARD_PATH}/code/databoard/tools/celery_info
chmod 777 ${DATABOARD_PATH}/code/databoard/tools/celery_info
# chmod 777 ${DATABOARD_PATH}/code/databoard/tools/celery_worker.sh
chmod a+r -R $DATABOARD_PATH/datacamp/databoard/problems
chmod a+r -R $DATABOARD_PATH/datacamp/databoard/submissions
# chmod a+x -R $DATABOARD_PATH/datacamp/databoard/submissions
# sudo -su ubuntu <<HERE
# echo Starting $NB_WORKERS workers as ubuntu user
# bash ${DATABOARD_PATH}/code/databoard/tools/celery_worker.sh start $NB_WORKERS
# HERE
cp ${DATABOARD_PATH}/code/databoard/tools/local_supervisord.conf ${DATABOARD_PATH}/code/databoard/tools/supervisord.conf
cp ${DATABOARD_PATH}/code/databoard/tools/local_celeryd.conf ${DATABOARD_PATH}/code/databoard/tools/celeryd.conf
cp ${DATABOARD_PATH}/code/databoard/tools/local_celerybeat.conf ${DATABOARD_PATH}/code/databoard/tools/celerybeat.conf
sed -i "s#DATABOARD_PATH#${DATABOARD_PATH}#g" ${DATABOARD_PATH}/code/databoard/tools/supervisord.conf
sed -i "s#DATABOARD_PATH#${DATABOARD_PATH}#g" ${DATABOARD_PATH}/code/databoard/tools/celeryd.conf
sed -i "s#DATABOARD_PATH#${DATABOARD_PATH}#g" ${DATABOARD_PATH}/code/databoard/tools/celerybeat.conf
easy_install supervisor
supervisord -c ${DATABOARD_PATH}/code/databoard/tools/supervisord.conf

# Add backup for the production server:
# execution of tools/dump_db.sh and tools/housekeeping.sh with crontab
# Not executed in the script because backup path must be checked and no backup for
# test server
# Add these lines to the file opened by crontab -e
# 02 0    * * *   root    bash /mnt/ramp_data/code/databoard/tools/dump_db.sh >/dev/null 2>&1
# 22 1    * * *   root    bash /mnt/ramp_data/code/databoard/tools/housekeeping.sh >/dev/null 2>&1

