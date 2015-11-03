#!/bin/sh

# source .bashrc
# ami-2cbf3e44

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

# echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/dist-packages/' >> .bashrc
# echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages/' >> .bashrc
# source .bashrc

git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
sudo cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF .
sudo make -j 4
sudo -H pip install .
sudo python setup.py build
sudo python setup.py install
cd ..


# INSTALL NVIDIA DRIVERS

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get -y install build-essential

wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run

chmod +x cuda_7.5.18_linux.run
mkdir nvidia_installers
sudo ./cuda_7.5.18_linux.run -extract=`pwd`/nvidia_installers

sudo apt-get -y install linux-image-extra-virtual

echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u

sudo apt-get -y install linux-source
sudo apt-get -y install linux-headers-`uname -r`
sudo apt-get -y install linux-headers-generic

sudo reboot


cd nvidia_installers
sudo ./NVIDIA-Linux-x86_64-352.39.run

sudo modprobe nvidia
sudo apt-get install build-essential

sudo ./cuda-linux64-rel-7.5.18-19867135.run
sudo ./cuda-samples-linux-7.5.18-19867135.run

echo "export PATH=/usr/local/cuda-7.5/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

source ~/.bashrc

cd


# NOW ENTER 'SUDO SU' - to become a superuser while not reseting the path
# RUN ALL THE COMMANDS YOU NEED

# sudo ln -s /usr/local/cuda-7.5/lib64/libcublas.so /usr/local/cuda-7.5/lib64/libcublas.so.7.5

# https://devtalk.nvidia.com/default/topic/845363/libcublas-so-7-0-cannot-open-shared-object-file/
# echo "/usr/local/cuda-7.5/lib" | sudo tee -a /etc/ld.so.conf.d/cuda.conf
# sudo ldconfig

# tar -zxf cudnn-6.5-linux-x64-v2.tgz
# cd cudnn-6.5-linux-x64-v2
# sudo cp lib* /usr/local/cuda-6.5/lib64/
# sudo cp cudnn.h /usr/local/cuda-6.5/include/



: ' 
sudo apt-get update -y
sudo apt-get install -y git wget linux-image-generic build-essential unzip

cd /tmp
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb

sudo apt-get update
sudo apt-get install -y cuda  

echo -e "\nexport CUDA_HOME=/usr/local/cuda-7.5\nexport CUDA_ROOT=/usr/local/cuda-7.5" >> ~/.bashrc
echo -e "\nexport PATH=/usr/local/cuda-7.5/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

source .bashrc

apt-get -y install linux-headers-generic

wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.55/NVIDIA-Linux-x86_64-352.55.run
sudo bash ./NVIDIA-Linux-x86_64-352.55.run

sudo apt-get -y install libcuda1-352
sudo apt-add-repository ppa:xorg-edgers/ppa && sudo apt-get update
sudo apt-get -y install nvidia-352 nvidia-352-dev nvidia-352-uvm libcuda1-352 nvidia-libopencl1-352 nvidia-icd-352

sudo apt-get update && sudo apt-get install cuda

sudo reboot
'

