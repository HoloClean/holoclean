# HoloClean: A Machine Learning System for Data Enrichment

[HoloClean](www.holoclean.io) is build ontop of PyTorch and Postgres.

HoloClean is a statistical inference engine to impute, clean, and enrich data.
As a weakly supervised machine learning system, HoloClean leverages available
quality rules, value correlations, reference data, and multiple other signals
to build a probabilistic model that accurately captures the data generation
process, and uses the model in a variety of data curation tasks. HoloClean
allows data practitioners and scientists to save the enormous time they spend
in building piecemeal cleaning solutions, and instead, effectively communicate
their domain knowledge in a declarative way to enable accurate analytics,
predictions, and insights form noisy, incomplete, and erroneous data.

## Installation

### 1. Install Postgres

#### Ubuntu

Install Postgres by running
```sh
apt-get install postgresql postgresql-contrib
```

#### MacOS

Installation instructions can be found at
[https://www.postgresql.org/download/macosx/](https://www.postgresql.org/download/macosx/).

### 2. Setup Postgres for HoloClean

To start the Postgres console from your terminal
```sh
psql --user <username>      # or you can omit --user <username> to use current user
```

We then create a database `holo` and user `holocleanuser` (default settings for HoloClean)
```
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

In general, to connect to the `holo` database from the Postgres console
```
\c holo
```

HoloClean currently populates the database `holo` with auxiliary and meta tables.
To clear the database simply connect as a root user or as `holocleanuser` and run
```
DROP DATABASE holo;
CREATE DATABASE holo;
```

### 3. Install HoloClean

#### Option 1: pip and conda (recommended)

##### Ubuntu

For **32-bit machines** run
```sh
wget https://repo.continuum.io/archive/Anaconda-2.3.0-Linux-x86.sh
sh Anaconda-2.3.0-Linux-x86.sh
```

For **64-bit machines** run
```
wget https://repo.continuum.io/archive/Anaconda-2.3.0-Linux-x86_64.sh
sh Anaconda-2.3.0-Linux-x86_64.sh
```

##### MacOS

Follow instructions [here](https://conda.io/docs/user-guide/install/macos.html) to install
Anaconda (NOT miniconda).

##### Create a conda environment

Create a **Python 3** conda environment by running
```sh
conda create -n holo_env python=3
```

Upon starting/restarting your terminal session, you will need to activate your
conda environment by running
```sh
source activate holo_env
```

**NOTE: ensure your environment is activated throughout the installation process.**

#### Install required packages
```sh
pip install -r requirements.txt
```

##### Install the holoclean package

Install `holoclean` via `pip`
```
pip install holoclean
```

#### Option 2: pip and Virtualenv

If you are familiar with Virtualenv, create a new **Python 3** environment
with your preferred Virtualenv wrapper, for example:

- [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) (Bourne-shells)
- [virtualfish](https://virtualfish.readthedocs.io/en/latest/) (fish-shell)

##### Install Virtualenv

Either follow instructions [here](https://virtualenv.pypa.io/en/stable/installation/) or install via
`pip`
```sh
pip install virtualenv
```

##### Create a Virtualenv environment

Create a new directory for a **Python 3** virtualenv environment
```sh
mkdir -p holo_env
virtualenv --python=python holo_env
```
where `python` is a valid reference to a python executable.

Activate the environment
```
source bin/activate
```

**NOTE: ensure your environment is activated throughout the installation process.**

##### Install the holoclean package

Install `holoclean` via `pip`
```
pip install holoclean
```

#### Option 3: Manual (from source)

You can manually clone this repository
```sh
git clone git@github.com:HoloClean/holoclean.git
cd holoclean
```

It is recommended you still create a conda or Virtualenv environment before
installing the package below (see above for instructions for creating either
types of environment). Install the `holoclean` package using `setuptools` by running
```sh
python setup.py install
```

## Usage

After installation, you can use `holoclean` as a standalone Python module
```python
import holoclean

### 0. Setup holoclean session

hc = holoclean.HoloClean()
session = hc.session

### 1. Load training data and denial constraints

# 'hospital' is the name of your dataset
# 'data' is the path to the CSV file
# 'hospital.csv' is the CSV filename
session.load_data('hospital', 'data', 'hospital.csv')
# Denial constraints in a TXT file
session.load_dcs('data','hospital_constraints_att.txt')
session.ds.set_constraints(session.get_dcs())

### 2. Detect error cells

detectors = [NullDetector(),ViolationDetector()]
hc.detect_errors(detectors)

### 3. Repair errors

hc.setup_domain()
featurizers = [InitFeaturizer(),OccurFeaturizer(), ConstraintFeat()]
hc.repair_errors(featurizers)

### 4. Evaluate results

# 'hospital_clean.csv' is the ground truth (i.e. test set labels)
hc.evaluate('data','hospital_clean.csv', get_tid, get_attr, get_value)
```


## Contributing (advanced)

### Setting up development environment

It is recommended you create a conda environment when developing (see installation
instructions above for conda).

1. Create a conda environment for developing holoclean
    ```sh
    conda create -n holo_dev python=3
    ```

2. Activate your environment (**must do this every time you start/restart a new terminal session**):
    ```sh
    source activate holo_dev
    ```

3. Install `holoclean` as a local editable package
    ```sh
    python setup.py develop
    ```

4. Verify that you've installed it
    ```sh
    > conda list | grep holoclean
    holoclean                 0.2.0                     <pip>
    ```

5. You should be able to import `holoclean` from anywhere now!
    ```sh
    python -c "import holoclean"
    ```

### Testing

After setting up your development environment and setting up `holoclean` as a
development package, you should be able to run any of the tests under
`tests/`, for example
```sh
sh tests/start_test.sh
```

### Building as a conda package

To build Holoclean as a conda package, first install `conda-build`
```
conda install conda-build
```
add the `pytorch` and `conda-forge` channels to your conda config
(`~/.condarc`) if you haven't already done so
```
conda config --add channels conda-forge
conda config --add channels pytorch
```
then run the following command in the terminal in this repository:
```sh
conda-build .
```

