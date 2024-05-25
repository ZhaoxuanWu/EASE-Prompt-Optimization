# Data
Part of the data is from the [Instruction Induction](https://arxiv.org/abs/2205.10782) dataset.
The datasets could be downloaded either from the GitHub repository of [Big-bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks) or [InstructZero](https://github.com/Lichang-Chen/InstructZero/tree/main/InstructZero/experiments/data/instruction_induction).

Code for processing the datasets to our experimental setting is in [data_process.ipynb](data_process.ipynb).


## Content

- raw:
	- induce: comprises input-output demonstrations that were utilized in the creation of the instruction induction inputs.
	- valid: comprises the validation set.
	- execute: examples that were withheld for evaluating the accuracy of execution.