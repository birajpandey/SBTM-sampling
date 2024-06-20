# Score Based Transport Modeling (SBTM) for sampling 

In this repo, we will modify SBTM method from Boffi et. al. 2023 for 
sampling from complex probability measures. The authors of this repo are 
[@birajpandey](https://github.com/birajpandey) and [@Vilin97](https://github.com/Vilin97). 


## Setup:

1. Do a clean download of the repository.
   ```
   git clone https://github.com/birajpandey/SBTM-sampling.git
   ```
2. Go to the downloaded repo
   ```
   cd path/to/SBTM-sampling
   ```
3. Run the `Makefile`. It creates an anaconda environment called `sbtm_env`, 
   downloads  required  packages, datasets and runs tests.

   ```
   make 
   ```
3. Activate the conda environment. 

   ```
   conda activate sbtm_env
   ```

4. Install the `sbtm` package
   ```
   pip install -e .
   ```
   
5. Run the files in `scripts/` to reproduce our results. 

__Remark__: This project structure is based on the 
<a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">
cookiecutter data science project template</a>. We also took a great deal of 
help from the <a target="_blank" href="https://goodresearch.dev/#alternative-formats">
The Good Research Code Handbook</a> written by Patrick J Mineault. 