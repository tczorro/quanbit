#!/usr/bin/env python
# Copyright (c) 2020, Xiaotian Derrick Yang
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Package build and install script."""


from setuptools import find_packages, setup


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.md") as f:
        return f.read()


setup(
    name="quanbit",
    version="0.0.1",
    description="Python library for simulating quantum computor and algorithm.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Xiaotian Derrick Yang",
    author_email="yxt1991@gmail.com",
    url="https://github.com/tczorro/quanbit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    install_requires=[
        "numpy>=1.16",
        "pytest>=2.6",
    ],
)
