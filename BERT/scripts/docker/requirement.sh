#!/bin/bash 

set -xe
#install cuda
apt install libxml2 -y
#wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run requirement/
sh requirements/cuda_10.2.89_440.33.01_linux.run

#install nsys
#wget 
dpkg -i requirements/NVIDIA_Nsight_Systems_Linux_CLI_Only_2020.3.1.72.deb


#install cupy
pip install cupy-cuda102


echo "export PATH=$PATH:/usr/local/cuda-10.2/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" >> ~/.bashrc
echo "alias nsys=/opt/nvidia/nsight-systems-cli/2020.3.1/target-linux-x64/nsys" >> ~/.bashrc
#source ~/.bashrc