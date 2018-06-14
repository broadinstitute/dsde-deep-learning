#!/bin/bash
ECHO=echo
VM_NAME="ddddsde-deep-learner-$USER"
$ECHO echo "Try to create vm named ${VM_NAME}"
$ECHO gcloud compute instances create "${VM_NAME}" \
	--scopes "compute-rw,storage-full,cloud-platform" \
	--image-family "ubuntu-1604-lts" --image-project "ubuntu-os-cloud" \
	--machine-type "n1-standard-4" \
	--accelerator "type=nvidia-tesla-k80,count=1" \
	--boot-disk-size "100" --boot-disk-type "pd-ssd" \
	--zone "us-east1-d" \
	--maintenance-policy TERMINATE --restart-on-failure

# Now try to SSH into the instance
$ECHO gcloud compute ssh "${VM_NAME}" --zone us-east1-d


$ECHO "echo To copy things from the VM to a local machine use gcloud compute scp "
$ECHO "echo For example the templates templates learned and plotted by keras_example.py can be copied locally with:"
$ECHO gcloud compute scp ${VM_NAME}:~/dsde-deep-learning/api_tutorials/frames/mnist/regression/templates_* ./ --zone us-east1-d



