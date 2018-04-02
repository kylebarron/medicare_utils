#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [
    'fastparquet >= 0.1.4',
    'joblib >= 0.11',
    'numpy >= 1.14.1',
    'pandas >= 0.22.0',
    'pyarrow >= 0.9.0',
    'requests >= 2.18.4',
    'requests-html >= 0.9.0',
    'tqdm >= 4.19.9',
]

setup_requirements = [
    'setuptools>=38.6.0'
]

test_requirements = []

setup(
    author="Kyle Barron",
    author_email='barronk@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Scripts to assist working with Medicare data.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='medicare_utils',
    name='medicare_utils',
    packages=find_packages(include=['medicare_utils']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/kylebarron/medicare_utils',
    version='0.0.1',
    zip_safe=False,
)
