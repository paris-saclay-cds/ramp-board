#!/bin/zsh
# :Usage: bash deploy_databoard.sh
# Prepare Ubuntu (14.04) server instance for the application deployment 
# We follow steps (+ other steps) from 
# - https://www.digitalocean.com/community/tutorials/how-to-serve-django-applications-with-apache-and-mod_wsgi-on-ubuntu-14-04

# Add environment variables
# env.sh file with environment variables must be in the same folder as this script
mv env.sh ~/.aliases
sed -i -e '$a source ~/.aliases' ~/.zshrc
source ~/.zshrc

# Set locales variables
export LC_ALL=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

cd /home/

# Install Packages from the Ubuntu Repositories 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" update 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install python-pip apache2 libapache2-mod-wsgi
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install git
# wget https://raw.github.com/brainsik/virtualenv-burrito/master/virtualenv-burrito.sh
# bash virtualenv-burrito.sh 
# source /root/.venvburrito/startup.sh

# Mount ScienceFS disk
apt-get -y install sshfs
mkdir /mnt/datacamp
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp

# Clone the project
git clone -b postgres2 --single-branch https://camille24@bitbucket.org/kegl/databoard.git
cd databoard

# Install Postgres
sudo apt-get -y install python-dev libpq-dev postgresql postgresql-contrib
pg_createcluster 9.3 main --start
# Change postgres permissions
sed -i "85c local   all             postgres                                trust" /etc/postgresql/9.3/main/pg_hba.conf 
sudo service postgresql restart
psql -U postgres -c '\i tools/setup_database.sql'
# Change database user permissions
sed -i "86i local   all             $DATABOARD_DB_USER                                 trust" /etc/postgresql/9.3/main/pg_hba.conf
sudo service postgresql restart

# Configure a Python Virtual Environment
# mkvirtualenv datarun
sudo apt-get -y install python-numpy python-scipy  
pip install -Ur requirements.txt
python setup.py develop

# Reacreate the database
python -c 'from databoard import db; db.create_all()'
# TODO add script here to recreate existing db 

# Configure Apache: copy apache conf file to /etc/apache2/sites-available/
mv /etc/apache2/sites-available/000-default.conf /etc/apache2/sites-available/backup.conf
cp tools/databoard.conf /etc/apache2/sites-available/000-default.conf

# Deal with environment variables
sed 's/=/ /g' ~/.aliases > tt.txt
sed "s/'//g" tt.txt > tt1.txt
sed "s/export/SetEnv/g" tt1.txt > tt.txt
while read p; 
do  
  sed -i "4a $p" /etc/apache2/sites-available/000-default.conf; 
done < tt.txt
sed -i "s/SetEnv/    SetEnv/g" /etc/apache2/sites-available/000-default.conf
rm tt.txt tt1.txt

# Wrapping up some permissions issues
# I don t think we need it, since nothing has to be written in the project dir
sudo chown -R www-data:www-data ../databoard
sudo chown :www-data ../.
# sudo chmod -R 777 $DIR_DATA
# sudo chmod -R 777 $DIR_SUBMISSION

# Restart Apache
sudo service apache2 restart


