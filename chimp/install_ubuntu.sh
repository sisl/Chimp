#!/bin/sh

sudo apt-get -y update && sudo apt-get -y upgrade

sudo apt-get -y install build-essential curl git m4 ruby texinfo libbz2-dev libcurl4-openssl-dev libexpat-dev libncurses-dev zlib1g-dev

sudo apt-get -y install libffi-dev libssl-dev
sudo apt-get -y install python
sudo apt-get -y install python-pip
sudo apt-get -y install libhdf5-dev
sudo apt-get -y install cython
sudo apt-get -y install python-h5py
sudo apt-get -y install python-pygame
sudo apt-get -y install python-matplotlib
sudo apt-get -y install python-numpy
sudo apt-get -y install python-scipy
sudo apt-get -y install cmake

sudo pip install --upgrade pip
sudo pip install pillow
sudo pip install -U six
sudo pip install chainer

:'
sudo apt-get install openjdk-8-jre
sudo pip install greenlet
sudo pip install selenium
sudo pip install wheel
'

git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
sudo cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF .
sudo make -j 4
sudo -H pip install .
sudo python setup.py build
sudo python setup.py install
cd ..


# git clone -b dev https://github.com/mcmachado/Arcade-Learning-Environment/ ale_dev

# INSTALL NVIDIA DRIVERS

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get -y install build-essential

wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

chmod +x cuda_6.5.14_linux_64.run
mkdir nvidia_installers
sudo ./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers

sudo apt-get -y install linux-image-extra-virtual

echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u

sudo apt-get -y install linux-source
sudo apt-get -y install linux-headers-`uname -r`
sudo apt-get -y install linux-headers-generic

sudo reboot

cd nvidia_installers
sudo ./NVIDIA-Linux-x86_64-340.29.run

sudo modprobe nvidia
sudo apt-get install build-essential

sudo ./cuda-linux64-rel-6.5.14-18749181.run
sudo ./cuda-samples-linux-6.5.14-18745345.run

echo "export PATH=/usr/local/cuda-6.5/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

source ~/.bashrc

cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

cd
