Deep Learning Recipes and Experiments with DNA reads and variants.
==================================================================

Quickstart
----------

    python recipes.py train_ref_read_anno --data_dir ./data/g94982_tensors_chr1_channels_last/ --id my_model

### Setting up your environment

We recommend using [anaconda](https://conda.io/docs/user-guide/install/index.html) to handle your python environments. For CPU only libraries:

    conda env create -n gatk -f ./gatkcondaenv_cpu.yml

To use GPU:

    conda env create -n gatk -f ./gatkcondaenv_gpu.yml
