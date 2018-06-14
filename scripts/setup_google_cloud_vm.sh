#!/bin/bash
#
# This script prepares a GPU enabled Google cloud VM instance for deep learning
# It is meant to be run on the VM itself.  
# SSH into the VM with: gcloud compute ssh "${VM_NAME}" --zone us-east1-d
# This script is to be run after running the companion script create_google_cloud_vm.sh on a local machine
#
# June 2018
# Sam Friedman 
# sam@broadinstitute.org

ECHO=echo

# Get root
$ECHO sudo passwd

# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  $ECHO curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ECHO sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ECHO sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  $ECHO sudo apt-get update
  $ECHO sudo apt-get install cuda-9-0 -y
fi

$ECHO source ~/.profile

# Enable persistence mode
$ECHO sudo nvidia-smi -pm 1

# Now we need to get NVidia's CuDNN libraries copied over and installed
$ECHO "echo Signup as a NVIdia's developer and download the CuDNN libs from: https://developer.nvidia.com/cudnn"
$ECHO "echo Or copy them from GSA or your local machine with gcloud compute scp:"
$ECHO "echo gcloud compute scp --recurse /dsde/working/sam/scripts/cuda ${VM_NAME}:~ --zone us-east1-d"
$ECHO sudo cp cuda/include/* /usr/local/cuda/include/
$ECHO sudo cp cuda/lib64/* /usr/local/cuda/lib64/
$ECHO sudo chmod 644 /usr/local/cuda/include/cudnn.h
$ECHO sudo ldconfig /usr/local/cuda/lib64

# Get Anaconda
$ECHO wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
$ECHO bash Anaconda3-4.0.0-Linux-x86_64.sh
$ECHO source ~/.profile

# Create the conda environment for GATK Deep Learning with GPU
$ECHO conda env create -n gatk -f $HOME/dsde-deep-learning/envs/gatkcondaenv_gpu.yml

#Activate environment
$ECHO source activate gatk

# Run something
$ECHO python $HOME/dsde-deep-learning/api_tutorials/keras_example.py

$ECHO "echo To copy things from the VM to a local machine use gcloud compute scp "
$ECHO "echo For example the templates templates learned by keras_example.py can be copied and looked at locally with:"
$ECHO "echo gcloud compute scp ${VM_NAME}:~/dsde-deep-learning/api_tutorials/frames/mnist/regression/templates_* ./ --zone us-east1-d"



