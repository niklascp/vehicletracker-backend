#!/usr/bin/env python
from setuptools import setup, find_packages

PROJECT_NAME = "Vehicle Tracker"
PROJECT_PACKAGE_NAME = "vehicletracker"
PROJECT_LICENSE = "Apache License 2.0"
PROJECT_URL = "https://github.com/niklascp/vehicletracker-backend"

PROJECT_AUTHOR = "Niklas Christoffer Petersen"
PROJECT_EMAIL = "niklascp@gmail.com"

import vehicletracker.const as const

with open('README.md', 'r') as file:
    long_description = file.read()

with open('requirements.txt') as file:
    install_requires = [line.rstrip('\r\n') for line in file]

PACKAGES = find_packages(exclude=["tests", "tests.*"])

setup(
    name=PROJECT_PACKAGE_NAME,
    version=const.__version__,
    url=PROJECT_URL,
    #download_url=DOWNLOAD_URL,
    #project_urls=PROJECT_URLS,
    author=PROJECT_AUTHOR,
    author_email=PROJECT_EMAIL,
    packages=PACKAGES,
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    #python_requires=">={}".format(MIN_PY_VERSION),
    test_suite="tests",
    entry_points={"console_scripts": ["vehicletracker = vehicletracker.__main__:main"]},
)
