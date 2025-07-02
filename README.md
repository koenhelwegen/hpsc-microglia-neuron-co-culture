This repository contains the data and analysis scripts for the paper [_Human pluripotent stem cell-derived microglia shape neuronal morphology and enhance network activity in vitro_](https://www.sciencedirect.com/science/article/pii/S0165027024002991).

The raw data used for the analyses is provided in the file `data_hpsc_microglia_neuron_coculture.xlsx` that is part of this repository.

## Setup

All scripts were run using Python 3.11.

To clone and access this repo, run:

```
git clone https://github.com/koenhelwegen/hpsc-microglia-neuron-co-culture
cd hpsc-microglia-neuron-co-culture
```

### Option 1: use a virtual environment (recommended)

To install & run within a virtual environment, run:

```
python3 -m virtualenv venv
source ./venv/bin/activate
python3 -m pip install -r requirements.txt
```

You may need to install virtualenv first:

```
python3 -m pip install virtualenv
```

### Option 2: install requirements in main python environment

To install the requirements in your main python environment
(WARNING: this may change the versions of previously installed packages),
run:

```
python3 -m pip install -r requirements.txt
```

## Analysis

To perform the analysis, run:

```
python3 analysis.py
```