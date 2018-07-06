# Quick Start guide

## Importing the package

First, make sure you've [installed](installation.html) `medicare_utils`.
Then to use the package, you need to import it:

```py
import medicare_utils as med
```

The `as med` means that you can refer to the package as `med` instead of `medicare_utils` from here on.

## Data extracts

Data extracts are started with the `med.MedicareDF` function. For example, I can begin an extract using 1% sample data and for the years 2010-2012 with:
```py
mdf = med.MedicareDF(percent=1, years=range(2010, 2013))
```

Note that the `range` function includes integers up to but not including the second argument.

Then I can get a cohort of white women aged 66-75


In recent years, Python has become the `fastest growing major programming language <https://stackoverflow.blog/2017/09/06/incredible-growth-python/>`_, largely due to its widespread use among data scientists. This popularity has fostered packages that work with data, such as `Pandas <https://pandas.pydata.org/>`_, the standard for in-memory data analysis. A newer package, `Dask <https://dask.pydata.org/en/latest/dataframe.html>`_, has been developed to parallelize Pandas operations and work with data larger than memory.
