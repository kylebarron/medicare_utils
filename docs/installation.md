# Installation

**This package only supports Python 3.6 or higher.** You can find out the version of Python installed by running `python --version` in your terminal. The first two numbers must be 3.6 or 3.7.

```
$ python --version
Python 3.6.4 :: Anaconda custom (64-bit)
```

## Stable release

To install medicare_utils, run this command in your terminal:

```
$ pip install medicare_utils --upgrade
```

This is the preferred method to install medicare_utils, as it will always install the most recent stable release.

If you don't have [`pip`](https://pip.pypa.io) installed, I recommend installing the [Anaconda distribution](https://www.anaconda.com/download), which will install a wide variety of helpful data science packages.
Otherwise, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process of installing `pip` manually.

## Development version

If you want the newest version available, you can install direct from the Github repository with:
```
$ pip install git+https://github.com/kylebarron/medicare_utils --upgrade
```

## From sources

The sources for medicare_utils can be downloaded from the [Github repo](https://github.com/kylebarron/medicare_utils).

You can either clone the public repository:

```
$ git clone git://github.com/kylebarron/medicare_utils
```

Or download the [tarball](https://github.com/kylebarron/medicare_utils/tarball/master):

```
$ curl  -OL https://github.com/kylebarron/medicare_utils/tarball/master
```

Once you have a copy of the source, you can install it with:

```
$ python setup.py install
```
