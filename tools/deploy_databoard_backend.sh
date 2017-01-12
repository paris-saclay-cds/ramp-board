#!/bin/bash
# Prepare Ubuntu (14.04) server instance for the application deployment 
#
# :Usage: bash deploy_databoard_backend.sh 
# There must be a file env.sh containing all required environment variables
# in the same folder as this script:
#    export DATABOARD_IP='xx.xx.xx.xx'  # IP of the frontent
#    export DATABOARD_PATH='/mnt/ramp_data/'  # databoard on the frontend server is in $DATABOARD_PATH/datacamp/databoard
#    export DATABOARD_DB_NAME='databoard'
#    export DATABOARD_DB_USER='xxxx'
#    export DATABOARD_DB_PASSWORD='yyyy'
#    export DATABOARD_DB_URL='postgresql://xxxx:yyyy@localhost/databoard'
#
# Before running this, make sure that you can log in to frontend server as root


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
sudo python get-pip.py 
sudo pip install pyopenssl ndg-httpsclient pyasn1
# Install Ubuntu dependencies for Python (for packages that need compilation)
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install build-essential python-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install gfortran
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install swig
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libatlas-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install liblapack-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libfreetype6 libfreetype6-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libxft-dev
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install pandoc
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libffi-dev
# Install numpy, scipy, and ipython
sudo pip install numpy
sudo pip install scipy
sudo pip install ipython
# Install git 
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install git
# Install postgresql
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install libpq-dev postgresql postgresql-contrib postgresql-common libpq-dev
# Install xgboost
cd; git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install
cd
# Install python-netcdf4 (requires zlib, hdf5, and netCDF-C)
# Takes a long time, comment it out if you don't need it in your ramp
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

# Pull databoard code
sudo mkdir $DATABOARD_PATH
cd $DATABOARD_PATH
sudo mkdir code
cd code
sudo git clone https://github.com/paris-saclay-cds/databoard.git
cd databoard
sudo pip install -r requirements.txt
sudo python setup.py develop

# Mount the frontend disk (for /submissions and /problems, which are not in the db)
sudo apt-get -y install sshfs
sudo mkdir $DATABOARD_PATH/datacamp
sudo mkdir $DATABOARD_PATH/datacamp/databoard
sudo sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/home/ubuntu/.ssh/id_rsa root@$DATABOARD_IP:$DATABOARD_PATH/datacamp/databoard $DATABOARD_PATH/datacamp/databoard

# Wrapping up some permissions issues
sudo chown -R :www-data ../.
sudo chown -R www-data:www-data ${DATABOARD_PATH}/datacamp/databoard

# Install zsh, htop
cd
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
echo 'source ~/.aliases' >> ~/.zshrc
sed -i 's/plugins=(git)/plugins=(git cp tmux screen pip lol fabric)/g' ~/.zshrc
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install htop

