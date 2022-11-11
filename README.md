This is console interface for the HORDA approach presented in the elsevier paper.
Over the console interface it is possible to evaluate instance defined in json format via the HORDA with defined neural network.

# Environment preparation
Due to a lot of version of ML libraries it is necessary to install its separately.

## Conda
Create conda environment without GPU:
```bash
conda create -n schnn python=3.7
conda activate schnn
conda install --file conda_requirements.txt
pip install pymonad==2.4.0
```

## Pip
You need python in version 3.7.
Install requirements from requirements.txt file.

# Example of usage

## Instance defined over stdin
It is possible to give the instance over stdin during the evaluation of the python code. 
The code expect the line with processing times of instances and secondly the line with due dates of instances, separated by space.
It is possible to give more than one instance, for finish of input mode write -1 after submitting due dates.
`console.py --nn nn/best/nn.json.nn`

## Instance defined over the command line
It is possible to give one instance directly to the python code over the cli.
It is expected to give the instance in json format, as you can see in following example.
`console.py --nn nn/out/best/nn.json.nn --instance "{"proc":[90, 51, 2, 90, 37, 16, 96, 45, 60, 1, 99, 32, 86, 55, 29, 26, 96, 100, 9, 7],"due":[367, 460, 368, 371, 423, 403, 375, 409, 323, 466, 313, 488, 429, 352, 318, 440, 373, 315, 494, 491]}"`

## Instance defined by file with instances
It is possible to submit the file with more than one instance.
The file should be list of instances.
`console.py --nn nn/out/best/nn.json.nn --instances_file test_instances.json`

## Specify the path for store of the output
It is possible to store the results to the fie, with flag `--output_file`. 
For example `console.py --nn nn/out/best/nn.json.nn --instances_file test_instances.json --output_file out.json`.

## Get optimal solutions
Warning: This can be time and memory demanding!
You can use switch `--optimal` in these case, the code also compute the optimal solution by our method.
It is reimplementation of the `SDD` approach in `python`, however not time and memory efficient.
The `tkindt` solver is more power.
