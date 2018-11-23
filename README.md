# HoloClean: A Machine Learning System for Data Enrichment

[HoloClean](http://www.holoclean.io) is built on top of PyTorch and Postgres.

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
```bash
$ apt-get install postgresql postgresql-contrib
```

#### MacOS

Installation instructions can be found at
[https://www.postgresql.org/download/macosx/](https://www.postgresql.org/download/macosx/).

### 2. Setup Postgres for HoloClean

To start the Postgres console from your terminal
```bash
$ psql --user <username>    # you can omit --user <username> to use current user
```

We then create a database `holo` and user `holo` (default settings for HoloClean)
```
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

In general, to connect to the `holo` database from the Postgres psql console
```
\c holo
```

HoloClean currently populates the database `holo` with auxiliary and meta tables.
To clear the database simply connect as a root user or as `holocleanuser` and run
```
DROP DATABASE holo;
CREATE DATABASE holo;
```

### 3. Set up HoloClean

#### Virtual Environment

##### Option 1: Set up a conda Virtual Environment

Install Conda using one of the following methods

##### Ubuntu

For **32-bit machines** run
```bash
$ wget https://repo.continuum.io/archive/Anaconda-2.3.0-Linux-x86.sh
$ sh Anaconda-2.3.0-Linux-x86.sh
```

For **64-bit machines** run
```bash
$ wget https://repo.continuum.io/archive/Anaconda-2.3.0-Linux-x86_64.sh
$ sh Anaconda-2.3.0-Linux-x86_64.sh
```

##### MacOS

Follow instructions [here](https://conda.io/docs/user-guide/install/macos.html) to install
Anaconda (NOT miniconda).

##### Create a conda environment

Create a **Python 2.7** conda environment by running

```bash
$ conda create -n holo_env python=2.7
```

Upon starting/restarting your terminal session, you will need to activate your
conda environment by running
```bash
$ source activate holo_env
```

**NOTE: ensure your environment is activated throughout the installation process.**

##### Option 2: Set up a virtual environment using pip and Virtualenv

If you are familiar with Virtualenv, create a new **Python 2.7** environment
with your preferred Virtualenv wrapper, for example:

- [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) (Bourne-shells)
- [virtualfish](https://virtualfish.readthedocs.io/en/latest/) (fish-shell)

##### Install Virtualenv

Either follow instructions [here](https://virtualenv.pypa.io/en/stable/installation/) or install via
`pip`
```bash
$ pip install virtualenv
```

##### Create a Virtualenv environment

Create a new directory for a **Python 2.7** virtualenv environment
```bash
$ mkdir -p holo_env
$ virtualenv --python=python holo_env
```
where `python` is a valid reference to a python executable.

Activate the environment
```bash
$ source bin/activate
```

**NOTE: ensure your environment is activated throughout the installation process.**

##### Install the requirements of HoloClean

In the project root directory, run the following to install the required packages.
Note that this commands installs the packages within the activated virtual environment.

```bash
$ pip install -r requirements.txt
```

### Note about MacOS

If you are on MacOS, you may need to install XCode developer tools using the command `xcode-select --install`.


## Usage

See the code in `examples/holoclean_repair_example.py` for a documented usage of HoloClean.

In order to run the example script, run the following:
```bash
$ cd examples
$ ./start_example.sh
```

The script sets up the python path environment for running holoclean.
