# Quick Start guide

## Importing the package

First, make sure you've [installed](installation.md) `medicare_utils`.
Then to use the package, you need to import it with the following line:
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



