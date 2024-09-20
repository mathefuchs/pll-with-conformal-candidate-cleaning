# Partial-Label Learning with Conformal Candidate Cleaning

This repository contains the code and data of the paper

> T. Fuchs, F. Kalinke (2025); "Partial-Label Learning with Conformal Candidate Cleaning"; Conference on Uncertainty in Artificial Intelligence (UAI).

This document provides (1) an outline of the repository structure and (2) steps to reproduce the experiments including setting up a virtual environment.

## Repository Structure

* The folder `data` contains all datasets used within our work.
  * The subfolder `realworld-datasets` contains commonly used real-world datasets for partial-label learning, which were initially provided by [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/Resources.htm).
  * Pytorch will additionally download datasets within this folder during the first run of the experiments.
* The folder `experiments` is an initially empty folder that contains all experiments to evaluate. Run `python script_create_data.py` to populate it.
* The folder `models` contains code for supervised reference models such as the MLP architecture.
* The folder `partial_label_learning` contains the code for the experiments.
  * The subfolder `methods` contains all implementations of related-work algorithms and our method.
* The folder `results` contains the results of all experiments. Run `python script_run_all.py` to reevaluate all experiments.
* Additionally, there are the following files in the root directory:
  * `.gitignore`
  * `LICENSE` describes the repository's licensing.
  * `README.md` is this document.
  * `requirements.txt` is a list of all required `pip` packages for reproducibility.
  * `script_create_data.py` is a Python script to create all experimental configurations.
  * `script_run_all.py` runs all experimental configurations in the `experiments` folder on all algorithms.
  * `script_analyze_res.py` creates all LaTeX tables in the paper from the results in `results`.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all the necessary dependencies.
Our code is implemented in Python (version 3.13.1; other versions, including lower ones, might also work).

We used `venv` (version 20.29.1; other versions might also work) to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Next, you install `virtualenv` with

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m pip install virtualenv==20.29.1
```

</td>
<td>

``` powershell
python -m pip install virtualenv==20.29.1
```

</td>
</tr>
</table>

To create a virtual environment for this project, you have to clone this repository first.
Thereafter, change the working directory to this repository's root folder.
Run the following commands to create the virtual environment and install all necessary dependencies:

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
<td>

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
</tr>
</table>

## Reproducing the Experiments

Make sure that you created the virtual environment as stated above.
The script `script_create_data.py` creates all experimental data including the artificial noise.
The script `script_run_all.py` runs all the experiments.
Running all experiments takes roughly two to three day on a system with 48 cores and one `NVIDIA GeForce RTX 4090`.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
source venv/bin/activate
python script_create_data.py
python script_run_all.py
```

</td>
<td>

``` powershell
.\venv\Scripts\Activate.ps1
python script_create_data.py
python script_run_all.py
```

</td>
</tr>
</table>

This creates `.parquet.gz` files in `results` containing the results of all experiments.

## Using the Data

The experiments' results are compressed `.parquet` files.
You can easily read any of them with `pandas`.
The file names have the template `<method>_<dataset>_<seed>.parquet.gz`.

``` python
import pandas as pd

results = pd.read_parquet("results/conformal-proden_mnist_0.parquet.gz")
```

## Reproducing the Tables

To obtain tables from the data, use the Python script `script_analyze_res.py`.
Use the following snippets to generate all tables.
Generating all of them takes about a few seconds on a single core.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
source venv/bin/activate
python script_tables_and_plots.py
```

</td>
<td>

``` powershell
.\venv\Scripts\Activate.ps1
python script_tables_and_plots.py
```

</td>
</tr>
</table>
