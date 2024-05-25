This is the official code for the paper: 
> Prompt Optimization with EASE? Efficient Ordering-aware Automated Selection of Exemplars.
>
> Zhaoxuan Wu, Xiaoqiang Lin, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick Jaillet, Bryan Kian Hsiang Low

# Prepare the data
We have prepared the data and put the datasets in the folder [experiments/data](experiments/data).
More information about data processing can be found there as well.

# Run our code
To run our code, first install the required conda environment.
```bash
conda env create -f environment.yml
```

We provide the commands to reproduce results from our paper below.
```bash
bash experiments/run_template.sh
```

Note that the code for the baseline methods is also included in this repository. The commands to run them are also included in the `experiments/run_template_ucb.sh` file.