#!/bin/bash
# Before running this, make sure that you can log in to frontend server as root
# Add environment variables to env.sh, see the README
cp env.sh ~/.aliases
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

# Install xgboost
cd; git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install
cd
# Install tensorflow from linux binary
sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
# Install python-tk
sudo apt-get install python python-tk idle python-pmw python-imaging
# Install python-netcdf4 (requires zlib, hdf5, and netCDF-C)
# Takes a long time, comment it out if you don't need it in your ramp
function install_hdf5 {
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
}
install_hdf5

# Mount the frontend disk (for /submissions and /problems, which are not in the db)
sudo apt-get -y install sshfs
sudo mkdir $DATABOARD_PATH
sudo mkdir $DATABOARD_PATH/datacamp
sudo mkdir $DATABOARD_PATH/datacamp/databoard
sudo sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/home/ubuntu/.ssh/id_rsa root@$DATABOARD_IP:$DATABOARD_PATH/datacamp/databoard $DATABOARD_PATH/datacamp/databoard

# Install zsh, htop
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
echo 'source ~/.aliases' >> ~/.zshrc
sed -i 's/plugins=(git)/plugins=(git cp tmux screen pip lol fabric)/g' ~/.zshrc
sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install htop

