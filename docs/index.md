.. medicare_utils documentation master file, created by
   sphinx-quickstart on Mon Apr 16 21:31:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to medicare_utils's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


# Welcome to `medicare_utils`'s documentation!

Medicare_utils is a collection of scripts and data to make working with Medicare data easier.
This was originally developed for use on the National Bureau of Economic Research's servers, but portions of the package may be useful for third parties as well.

This package contains no Medicare data or private information. It assumes you already have access to Medicare data.

At this point, there are a few main pieces of code in this package:

- `MedicareDF` class. This makes it easy to create automated extracts of data based on different characteristics, and to automatically check for ICD-9 or HCPCS codes for a given subset of people.
- Classes to work with NPI, ICD-9, and HCPCS codes. These commands will automatically download these data files for you. [^copyright]
- `parquet` class. This provides a simple interface to convert files from Stata format to the modern Parquet format.

[^copyright]: The HCPCS codes
Datasets with HCPCS codes and short descriptions from 2003 to the present are freely available on the CMS website in their [Relative Value Files](https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/PhysicianFeeSched/PFS-Relative-Value-Files.html). These CMS files are released under the [End User Point and Click Agreement](https://www.cms.gov/apps/aha/license.asp?file=/Medicare/Medicare-). In order to not run afoul of this license agreement, the `medicare_utils` package does not distribute HCPCS codes. Rather, it provides code for the user to download and work with them. By using the HCPCS functions in this package, you agree to the above Agreement.


Don't worry if you don't know what a `class` is yet! This documentation aims to walk through everything needed to run these routines. Then you can keep working with these extracts in Python or easily export them to Stata's `.dta` format. Head to the [Quick Start guide](quickstart.md) to get started.
