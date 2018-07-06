Welcome to medicare_utils's documentation!
==========================================

A Python package to make working with Medicare data easier.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation.md
   quickstart.md
   usage.md
   api.rst
   authors.md
   contributing.md
   history.md

Introduction
------------

Medicare data are large and unwieldy. Since the size of the data is often larger than memory, many people use SAS. However SAS is an ugly language and not enjoyable to work with. Python is an easier and faster alternative.


Creating a data extract can be done in three lines of code:

.. code-block:: python

    import re
    import medicare_utils as med
    mdf = med.MedicareDF(
        percent='100',
        years=range(2008, 2014))
    mdf.get_cohort(
        gender='male',
        ages=range(65, 75),
        buyin_val=['3', 'C'],
        join='outer',
        keep_vars=['bene_dob'])
    mdf.search_for_codes(
        data_types=['med', 'opc'],
        icd9_dx=re.compile(r'^410'),
        icd9_dx_max_cols=1,
        collapse_codes=True,
        keep_vars={'med': ['medparid', 'admsndt', 'dschrgdt']},
        rename={'icd9_dx': 'ami'})

The resulting data extract consists of a patient-level file of patients who are:

- Male
- Aged 65-74 (inclusive) in any year from 2008 to 2013
- Continuously enrolled in fee-for-service Medicare in any year from 2008 to 2013 (i.e. :code:`buyin_val` either :code:`3` or :code:`C`)

and a claim-level file of patients who were included in the above cohort and furthermore had a primary diagnosis code of AMI in either the `MedPAR <https://kylebarron.github.io/medicare-documentation/resdac/medpar-rif/>`_ or `Outpatient claims <https://kylebarron.github.io/medicare-documentation/resdac/op-rif/>`_ files. The patient-level file is accessed with :code:`mdf.pl` and the claim-level file is accessed with :code:`mdf.cl`.

This package also provides:

- Classes to work with NPI, ICD-9, and HCPCS codes. These commands will automatically download these data files for you. [#copyright]_
- Codebooks for values of categorical variables.
- A simple interface to convert data files from Stata format to the modern Parquet format.

This documentation aims to walk through everything needed to run these routines. Then you can keep working with these extracts in Python or easily export them to Stata's :code:`.dta` format. Head to the :doc:`Quick Start guide </quickstart>` to get started.

Caveats
-------

This package contains no Medicare data or private information. It assumes you already have access to Medicare data.

This package was originally developed for use on the National Bureau of Economic
Research's servers, but portions of the package may be useful for third parties
as well.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: Footnotes

.. [#copyright] Datasets with HCPCS codes and short descriptions from 2003 to the present are freely available on the CMS website in their `Relative Value Files <https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/PhysicianFeeSched/PFS-Relative-Value-Files.html>`_. These CMS files are released under the `End User Point and Click Agreement <https://www.cms.gov/apps/aha/license.asp?file=/Medicare/Medicare->`_. In order to not run afoul of this license agreement, this package does not distribute HCPCS codes. Rather, it provides code for the user to download and work with them. By using the HCPCS functions in this package, you agree to the above Agreement.
