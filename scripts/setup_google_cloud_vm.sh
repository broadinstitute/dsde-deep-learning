#!/bin/bash
ECHO=echo
VM_NAME="dsde-deep-learner-$USER"
$ECHO echo "Try to create vm named ${VM_NAME}"
$ECHO gcloud compute instances create "${VM_NAME}" \
	--scopes "compute-rw,storage-full,cloud-platform" \
	--image-family "ubuntu-1604-lts" --image-project "ubuntu-os-cloud" \
	--machine-type "n1-standard-8" \
	--accelerator "type=nvidia-tesla-k80,count=1" \
	--boot-disk-size "500" --boot-disk-type "pd-ssd" \
	--zone "us-east1-d" \
	--maintenance-policy TERMINATE --restart-on-failure \
	--metadata startup-script='#!/bin/bash
    echo "Checking for CUDA and installing."
    # Check for CUDA and try to install.
    if ! dpkg-query -W cuda-9-0; then
      curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
      dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
      apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
      apt-get update
      apt-get install cuda-9-0 -y
    fi'



# Now try to SSH into the instance
$ECHO gcloud compute ssh "${VM_NAME}" --zone us-east1-d

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
$ECHO nvidia-smi -pm 1

$ECHO wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
$ECHO bash Anaconda3-4.0.0-Linux-x86_64.sh

$ECHO source ~/.profile

$ECHO git clone https://github.com/broadinstitute/dsde-deep-learning.git
$ECHO cd dsde-deep-learning/
$ECHO conda env create -n gatk -f ./envs/gatkcondaenv_gpu.yml

$ECHO source activate gatk



