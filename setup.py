# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

version_string = "1.0"

setup(
    name="holoclean",
    description=("A data imputation lib based on deep learning."),
    version=version_string,
    package_dir={"holoclean": "."},
    packages=["holoclean"] + ["holoclean." + package for package in find_packages(where=".")],
    include_package_data=True,
    python_requires="~=3.6",
    author="Apache Software Foundation",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    tests_require=["flask-testing==0.7.1"],
)
