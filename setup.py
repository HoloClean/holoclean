from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='holoclean',
    version='0.0.0',
    description='Holoclean is a statistical inference engine to impute, clean, and enrich data.',
    author='HoloClean',
    author_email='contact@holoclean.io',
    license='Apache License 2.0',
    url='http://www.holoclean.io/',
    packages=['holoclean'],
    install_requires=[
        'psycopg2==2.7.5',
        'torch==0.4.1',
        'python-Levenshtein==0.12.0',
        'sqlalchemy==1.2.12',
        'tqdm==4.15.0',
        'scipy==1.1.0',
        'numpy==1.15.3',
        'pandas==0.23.4',
    ]
)

