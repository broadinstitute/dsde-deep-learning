#!/bin/bash
ECHO=echo

# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  $ECHO curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ECHO dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ECHO apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  $ECHO apt-get update
  $ECHO apt-get install cuda-9-0 -y
fi
# Enable persistence mode
$ECHO sudo passwd
$ECHO nvidia-smi -pm 1

$ECHO wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
$ECHO bash Anaconda3-4.0.0-Linux-x86_64.sh

$ECHO source ~/.profile

$ECHO git clone https://github.com/broadinstitute/dsde-deep-learning.git
$ECHO cd dsde-deep-learning/
$ECHO conda env create -n gatk -f ./envs/gatkcondaenv_gpu.yml

$ECHO source activate gatk

$ECHO cd api_tutorials/
$ECHO python api_tutorials/keras_example.py

$ECHO "echo To copy things from the VM to a local machine use gcloud compute scp "
$ECHO "echo For example the templates templates learned by keras_example.py can be copied and looked at locally with:"
$ECHO "echo gcloud compute scp ${VM_NAME}:~/dsde-deep-learning/api_tutorials/frames/mnist/regression/templates_* ./ --zone us-east1-d"



