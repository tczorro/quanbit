#!/usr/bin/env python
# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
# pragma pylint: disable=invalid-name
"""Package build and install script."""


from setuptools import find_packages, setup

def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.md") as f:
        return f.read()

setup(
    name="quanbit",
    version="0.0.1",
    description="Python library for simulatin quantum computor ang algorithm.",
    # long_description=get_readme(),
    author="Derrick Yang",
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
