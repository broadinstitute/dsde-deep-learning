Deep Learning Recipes and Experiments with DNA reads and variants.
==================================================================

Quickstart
----------

    python recipes.py train_ref_read_anno --data_dir ./data/example_tensors/ --epochs 100 --patience 5

### Setting up your environment

We recommend using [anaconda](https://conda.io/docs/user-guide/install/index.html) to handle your python environments. On a mac:

    conda env create -n gatk -f ./gatkcondaenv_macosx.yml

On linux:

    conda env create -n gatk -f ./gatkcondaenv_linux.yml
