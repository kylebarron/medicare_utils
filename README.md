# medicare_utils

<!-- [![image](https://img.shields.io/pypi/v/medicare_utils.svg)](https://pypi.python.org/pypi/medicare_utils) -->

<!-- [![image](https://img.shields.io/travis/kylebarron/medicare_utils.svg)](https://travis-ci.org/kylebarron/medicare_utils) -->

<!-- [![Documentation Status](https://readthedocs.org/projects/medicare-utils/badge/?version=latest)](https://medicare-utils.readthedocs.io/en/latest/?badge=latest) -->

Scripts to assist working with Medicare data.

<!-- -   Free software: MIT license -->
<!-- -   Documentation: <https://medicare-utils.readthedocs.io>. -->

## Features

Provides the class `MedicareDF`. This class contains some canned scripts to make common tasks easier. It currently contains two functions:
- `get_cohort()`, which uses the beneficiary summary file to find a set of medicare beneficiaries according to options given to the function.
- `search_for_codes()`, which searches for HCPCS, ICD-9 diagnosis, and/or ICD-9 procedure codes in a given type of file.

## Installation

Install the package with:
```
pip install git+https://github.com/kylebarron/medicare_utils --upgrade
```

## Usage

The class is initialized with
```py
import medicare_utils as med
mdf = med.MedicareDF('05', range(2010, 2013))
mdf.get_cohort(gender='female', ages=range(65, 75))
mdf.search_for_codes(2010, 'med', icd9_diag='41071')
```

It has attributes that refer to different levels of the data, when applicable:
- `mdf.pl`: patient-level data. Here the index of the data is `bene_id` for data post-2005, or `ehic` for data pre-2005.
- `mdf.cl`: claim-level data.

