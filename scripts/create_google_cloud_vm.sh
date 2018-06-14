#!/bin/bash
#
# This script creates a GPU enabled Google cloud VM instance
# It is meant to be run on a local or GSA server machine
# After running this run the companion script setup_google_cloud_vm.sh on the instance
#
# June 2018
# Sam Friedman 
# sam@broadinstitute.org

ECHO=echo
ZONE=us-east1-d
VM_NAME="dsde-deep-learner-$USER"
BOOT_DISK_GB=200

# Create the VM with gcloud
$ECHO echo "Try to create vm named ${VM_NAME}"
$ECHO gcloud compute instances create "${VM_NAME}" \
	--scopes "compute-rw,storage-full,cloud-platform" \
	--image-family "ubuntu-1604-lts" --image-project "ubuntu-os-cloud" \
	--machine-type "n1-standard-4" \
	--accelerator "type=nvidia-tesla-k80,count=1" \
	--boot-disk-size "${BOOT_DISK_GB}" --boot-disk-type "pd-ssd" \
	--zone "${ZONE}" \
	--maintenance-policy TERMINATE --restart-on-failure

# We also need NVidia's CuDNN libraries
$ECHO "echo Signup as a NVIdia's developer and download the CuDNN libs from: https://developer.nvidia.com/cudnn"
$ECHO "echo Or if you have Broad filesystem acces copy them from GSA with gcloud compute scp:"
$ECHO gcloud compute scp --recurse /dsde/working/sam/scripts/cuda ${VM_NAME}:~ --zone $ZONE

# Now try to SSH into the instance
$ECHO gcloud compute ssh "${VM_NAME}" --zone us-east1-d

$ECHO "echo In a terminal on the instance clone this repo:"
$ECHO "echo git clone https://github.com/broadinstitute/dsde-deep-learning.git "
$ECHO "echo Then run the setup script with: "
$ECHO "echo ./dsde-deep-learning/scripts/setup_google_cloud_vm.sh "

$ECHO "echo To copy things from the VM to a local machine use gcloud compute scp "
$ECHO "echo For example the templates learned and plotted by keras_example.py can be copied locally with:"
$ECHO gcloud compute scp --recurse ${VM_NAME}:~/regression_example ./ --zone $ZONE
