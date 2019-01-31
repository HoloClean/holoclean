Master:
[![Build Status](https://travis-ci.org/HoloClean/holoclean.svg?branch=master)](https://travis-ci.org/HoloClean/holoclean)
Dev:
[![Build Status](https://travis-ci.org/HoloClean/holoclean.svg?branch=dev)](https://travis-ci.org/HoloClean/holoclean)

# HoloClean: A Machine Learning System for Data Enrichment

[HoloClean](http://www.holoclean.io) is built on top of PyTorch and PostgreSQL.

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

HoloClean was tested on Python versions 2.7, 3.6, and 3.7. 
It requires PostgreSQL version 9.4 or higher.


### 1. Install and configure PostgreSQL

We describe how to install PostgreSQL and configure it for HoloClean
(creating a database, a user, and setting the required permissions).

#### Option 1: Native installation of PostgreSQL

A native installation of PostgreSQL runs faster than docker containers.
We explain how to install PostgreSQL then how to configure it for HoloClean use.

##### a. Installing PostgreSQL

On Ubuntu, install PostgreSQL by running
`
$ apt-get install postgresql postgresql-contrib
`

For macOS, you can find the installation instructions on
[https://www.postgresql.org/download/macosx/](https://www.postgresql.org/download/macosx/)

##### b. Setting up PostgreSQL for HoloClean

By default, HoloClean needs a database `holo` and a user `holocleanuser` with permissions on it.

1. Start the PostgreSQL `psql` console from the terminal using \
`$ psql --user <username>`. You can omit `--user <username>` to use current user.

2. Create a database `holo` and user `holocleanuser`
```sql
CREATE DATABASE holo;
CREATE USER holocleanuser;
ALTER USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
```

You can connect to the `holo` database from the PostgreSQL `psql` console by running
`psql -U holocleanuser -W holo`.

HoloClean currently populates the database `holo` with auxiliary and meta tables.
To clear the database simply connect as a `root` user or as `holocleanuser` and run
```sql
DROP DATABASE holo;
CREATE DATABASE holo;
```

#### Option 2: Using Docker
If you are familiar with docker, an easy way to start using
HoloClean is to start a PostgreSQL docker container.

To start a PostgreSQL docker container, run the following command:

```bash
docker run --name pghc \
    -e POSTGRES_DB=holo -e POSTGRES_USER=holocleanuser -e POSTGRES_PASSWORD=abcd1234 \
    -p 5432:5432 \
    -d postgres:11
```

which starts a backend server and creates a database with the required permissions.

You can then use `docker start pghc` and `docker stop pghc` to start/stop the container.


Note the port number which may conflict with existing PostgreSQL servers.
Read more about this docker image [here](https://hub.docker.com/_/postgres/). 

### 2. Setting up HoloClean
HoloClean runs on Python 2.7 or 3.6+. We recommend running it from within
a virtual environment.

#### Creating a virtual environment for HoloClean
##### Option 1: Conda Virtual Environment

First, download Anaconda (not miniconda) from [this link](https://www.anaconda.com/download).
Follow the steps for your OS and framework. 

Second, create a conda environment (python 2.7 or 3.6+).
For example, to create a *Python 3.6* conda environment, run:

```bash
$ conda create -n hc36 python=3.6
```

Upon starting/restarting your terminal session, you will need to activate your
conda environment by running
```bash
$ conda activate hc36
```

##### Option 2: Set up a virtual environment using pip and Virtualenv

If you are familiar with `virtualenv`, you can use it to create 
a virtual environment.

For Python 3.6, create a new environment
with your preferred virtualenv wrapper, for example:

* [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) (Bourne-shells)
* [virtualfish](https://virtualfish.readthedocs.io/en/latest/) (fish-shell)


Either follow instructions [here](https://virtualenv.pypa.io/en/stable/installation/) or install via
`pip`.
```bash
$ pip install virtualenv
```

Then, create a `virtualenv` environment by creating a new directory for a Python 3.6 virtualenv environment
```bash
$ mkdir -p hc36
$ virtualenv --python=python3.6 hc36
```
where `python3.6` is a valid reference to a Python 3.6 executable.

Activate the environment
```bash
$ source hc36/bin/activate
```

#### Install the required python packages

*Note: make sure that the environment is activated throughout the installation process.
When you are done, deactivate it using* 
`conda deactivate`, `source deactivate`, *or* `deactivate` 
*depending on your version*.

In the project root directory, run the following to install the required packages.
Note that this commands installs the packages within the activated virtual environment.

```bash
$ pip install -r requirements.txt
```


*Note for macOS Users:*
you may need to install XCode developer tools using `xcode-select --install`.


## Running HoloClean

See the code in `examples/holoclean_repair_example.py` for a documented usage of HoloClean.

In order to run the example script, run the following:
```bash
$ cd examples
$ ./start_example.sh
```

Notice that the script sets up the Python path environment to run HoloClean.
