# MutCat
A Structural-Functional Basis for Functional Precision Medicine through Construction of Functional Mutational Catalogues


### Installation Notes ####

Clone Repository:
git clone https://github.com/ebrunk/MutCat.git

Create a conda environment:
cd MutCat
conda env create -f binder/environment.yml

Activate the conda environment:
conda activate MutCat

Launch Jupyter Notebook:
jupyter notebook

After you are finished, deactivate the conda environment:
conda deactivate

To permanently remove the benchmark environment:
conda remove -n MutCat --all


Setting Spark Configurations

When running PySpark on many cores (e.g., > 8), the memory for the Spark Driver and Workers may need to be increased. If necessary, set the environmental variable SPARK_CONF_DIR to the conf directory provided in this repository in your .bashrc (Linux) or .bash_profile (Mac) file.

export SPARK_CONF_DIR=<path>/MutCat/conf

Then close the terminal window and reopen it to set the environment variable.

The conf directory contains the file spark-env.sh with the following default settings.

SPARK_DRIVER_MEMORY=4G
SPARK_WORKER_MEMORY=4G

When running this repo on 24 core machine, you may need to increase the memory settings to 20G

