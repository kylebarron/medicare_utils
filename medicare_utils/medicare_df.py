#! /usr/bin/env python3
import re
import numpy as np
import pandas as pd
import fastparquet as fp
import dask.dataframe as dd
import pyarrow.parquet as pq

from time import time
from typing import Dict, List, NamedTuple, Optional, Pattern, Union
from collections import namedtuple
from multiprocessing import cpu_count

from .utils import fpath, _mywrap
from .codebook import codebook

allowed_pcts = ['0001', '01', '05', '20', '100']
pct_dict = {
    0.01: '0001',
    1: '01',
    5: '05',
    20: '20',
    100: '100',
    '0001': 0.01,
    '01': 1,
    '05': 5,
    '20': 20,
    '100': 100}


class MedicareDF(object):
    """Instantiate a MedicareDF object

    Attributes:
        percent (str, int, or float): percent sample of data to use
        years (list[int]): years of data to use
        year_type (str): ``calendar`` to work with multiple years as calendar
            years; ``age`` to work with patients' age years
        verbose (bool): Print progress status of program to console
        parquet_engine (str): ``pyarrow`` or ``fastparquet``
        parquet_nthreads (int): number of threads to use when reading file
        dta_path (str): path to Stata Medicare files
        pq_path (str): path to Parquet Medicare files
    Returns:
        ``MedicareDF`` object. Can then create a cohort with :func:`get_cohort`
        or search for patients with a given diagnosis with
        :func:`search_for_codes`
    """

    def __init__(
            self,
            percent: Union[str, int, float],
            years: Union[int, List[int]],
            year_type: str = 'calendar',
            verbose: bool = False,
            parquet_engine: str = 'pyarrow',
            parquet_nthreads: Optional[int] = None,
            dta_path: str = '/disk/aging/medicare/data',
            pq_path:
            str = '/disk/agebulk3/medicare.work/doyle-dua51929/barronk-dua51929/raw/pq'
            ) -> None:
        """Instantiate a MedicareDF object

        Attributes:
            percent: percent sample of data to use
            years: years of data to use
            year_type: ``calendar`` to work with multiple years as
                calendar years; ``age`` to work with patients' age years
            verbose: Print progress status of program to console
            parquet_engine: ``pyarrow`` or ``fastparquet``
            parquet_nthreads: number of threads to use when reading file
            dta_path: path to Stata Medicare files
            pq_path: path to Parquet Medicare files
        Returns:
            ``MedicareDF`` object. Can then create a cohort with
            :func:`get_cohort` or search for patients with a given diagnosis
            with :func:`search_for_codes`
        """

        # Check types
        if (type(percent) == float) or (type(percent) == int):
            try:
                self.percent = pct_dict[percent]
            except KeyError:
                msg = f"""\
                percent provided is not valid.
                Valid arguments are: {list(pct_dict.keys())}
                """
                raise ValueError(_mywrap(msg))
        elif type(percent) == str:
            if percent not in allowed_pcts:
                msg = f'percent must be one of: {allowed_pcts}'
                raise ValueError(msg)

            self.percent = percent
        else:
            raise TypeError('percent must be str or number')

        if type(years) == int:
            years = [years]
        else:
            years = years

        if (len(years) == 1) & (year_type == 'age'):
            msg = "year_type can't be `age` when only one year is given"
            raise ValueError(msg)

        self.years = years
        self.year_type = year_type
        self.verbose = verbose

        if parquet_engine not in ['pyarrow', 'fastparquet']:
            raise ValueError('parquet_engine must be pyarrow or fastparquet')

        if parquet_nthreads is None:
            parquet_nthreads = cpu_count()

        self.parquet_engine = parquet_engine
        self.parquet_nthreads = parquet_nthreads

        self.pl = None
        self.cl = None

        self.dta_path = dta_path
        self.pq_path = pq_path

    def _fpath(
            self, percent: str, year: int, data_type: str,
            dta: bool = False) -> str:

        return fpath(
            percent=percent,
            year=year,
            data_type=data_type,
            dta=dta,
            dta_path=self.dta_path,
            pq_path=self.pq_path)

    @staticmethod
    def _get_cohort_type_check(
            gender: Optional[str], ages: Union[int, List[int], None],
            races: Union[str, List[str], None], rti_race: bool,
            buyin_val: Union[str, List[str], None],
            hmo_val: Union[str, List[str], None], join: str,
            keep_vars: List[str], dask: bool, verbose: bool):
        """Check types and valid values for :func:`get_cohort`

        Also resolves input into correct value
        """

        # Check gender
        if gender is None:
            pass
        else:
            try:
                gender = str(int(gender))
            except ValueError:
                gender_cbk = codebook('bsfab')['sex']['values']
                gender_cbk = {v.lower(): k for k, v in gender_cbk.items()}
                gender_cbk = {
                    **gender_cbk,
                    **{k[0]: v
                       for k, v in gender_cbk.items()}}
                try:
                    gender = gender_cbk[gender.lower()]
                except KeyError:
                    raise ValueError(f'{gender} is invalid value for `gender`')

        # Check ages
        if type(ages) == int:
            ages = [ages]

        # check races
        assert (rti_race is None) | (type(rti_race) is bool)
        race_col = 'rti_race_cd' if rti_race else 'race'

        if races is None:
            pass
        elif type(races) == list:
            try:
                races = [str(int(x)) for x in races]
            except ValueError:
                race_cbk = codebook('bsfab')[race_col]['values']
                race_cbk = {v.lower(): k for k, v in race_cbk.items()}
                if race_col == 'rti_race_cd':
                    race_cbk['white'] = race_cbk.pop('non-hispanic white')

                races_new = []
                for race in races:
                    r = [v for k, v in race_cbk.items() if re.search(race, k)]
                    msg = f'`{race}` matches more than one race description'
                    assert len(r) <= 1, msg
                    msg = f'`{race}` matches no race description'
                    assert len(r) >= 1, msg
                    races_new.extend(r)

                races = races_new
        else:
            try:
                races = [str(int(races))]
            except ValueError:
                race_cbk = codebook('bsfab')[race_col]['values']
                race_cbk = {v.lower(): k for k, v in race_cbk.items()}
                if race_col == 'rti_race_cd':
                    race_cbk['white'] = race_cbk.pop('non-hispanic white')

                r = [v for k, v in race_cbk.items() if re.search(races, k)]
                msg = f'`{race}` matches more than one race description'
                assert len(r) <= 1, msg
                msg = f'`{race}` matches no race description'
                assert len(r) >= 1, msg

                races = r

        buyin_val = [buyin_val] if type(buyin_val) == str else buyin_val
        hmo_val = [hmo_val] if type(hmo_val) == str else hmo_val

        allowed_join = ['left', 'right', 'inner', 'outer']
        if join not in allowed_join:
            msg = f'join must be one of: {allowed_join}'
            raise ValueError(msg)

        keep_vars = [keep_vars] if type(keep_vars) == str else keep_vars

        assert type(dask) == bool
        assert type(verbose) == bool

        class Return(NamedTuple):
            gender: Optional[str]
            ages: Optional[List[int]]
            races: Optional[List[str]]
            rti_race: bool
            race_col: str
            buyin_val: Optional[List[str]]
            hmo_val: Optional[List[str]]
            join: str
            keep_vars: List[str]
            dask: bool
            verbose: bool

        return Return(
            gender=gender,
            ages=ages,
            races=races,
            rti_race=rti_race,
            race_col=race_col,
            buyin_val=buyin_val,
            hmo_val=hmo_val,
            join=join,
            keep_vars=keep_vars,
            dask=dask,
            verbose=verbose)

    def get_cohort(
            self,
            gender: Optional[str] = None,
            ages: Union[int, List[int], None] = None,
            races: Union[str, List[str], None] = None,
            rti_race: bool = False,
            buyin_val: Union[str, List[str], None] = None,
            hmo_val: Union[str, List[str], None] = None,
            join: str = 'outer',
            keep_vars: List[str] = [],
            dask: bool = False,
            verbose: bool = False):
        """Get cohort given demographic and enrollment characteristics

        Creates ``.pl`` attribute with patient-level data in the form of a
        pandas DataFrame. Index of returned DataFrame is always ``bene_id``,
        even if years provided are before 2006. In pre-2006 years, ``ehic`` will
        always be returned as a column.

        Args:
            gender: ``M``, ``F``, ``Male``, ``Female``, or ``None`` (keep both)
            ages:
                Range of ages to include. When ``year_type`` is ``calendar``,
                keeps anyone whose age was in ``ages`` at the end of the
                calendar year. When ``year_type`` is ``age``, keeps anyone whose
                age was in ``ages`` at any point during the year.
            races: Races to include
            rti_race: Whether to use the Research Triangle
                Institute race code instead of the default race code.
            buyin_val: The values ``buyinXX`` can take
            hmo_val: The values ``hmoindXX`` can take
            join: method for joining across years. Options: ``outer``,
                ``inner``, ``left``, ``right``. ``outer`` keeps all patients who
                matched desired characteristics in **any** year. ``inner`` keeps
                all patients who matched desired characteristics in **all**
                years. ``left`` keeps all people who matched desired
                characteristics in the **first** year. ``right`` keeps all
                people who matched desired characteristics in the **last** year.
            keep_vars: Variable names to keep in final output
            dask: Use dask library for out of core computation
            verbose: Print progress of program to console

        Returns:
            Creates attributes ``.pl`` with patient-level data in pandas
            DataFrame and ``.nobs_dropped`` with dict of percent of observations
            dropped due to each filter.
        """

        if self.verbose:
            verbose = True

        objs = self._get_cohort_type_check(
            gender=gender,
            ages=ages,
            races=races,
            rti_race=rti_race,
            buyin_val=buyin_val,
            hmo_val=hmo_val,
            join=join,
            keep_vars=keep_vars,
            dask=dask,
            verbose=verbose)
        gender = objs.gender
        ages = objs.ages
        races = objs.races
        rti_race = objs.rti_race
        race_col = objs.race_col
        buyin_val = objs.buyin_val
        hmo_val = objs.hmo_val
        join = objs.join
        keep_vars = objs.keep_vars
        dask = objs.dask
        verbose = objs.verbose

        if verbose:
            t0 = time()
            msg = f"""\
            Starting cohort retrieval
            - percent sample: {self.percent}
            - years: {list(self.years)}
            - ages: {list(ages) if ages else None}
            - races: {races if races else None}
            - buyin values: {buyin_val}
            - HMO values: {hmo_val}
            - extra variables: {keep_vars}
            """
            print(_mywrap(msg))

        # Get list of variables to import for each year
        if ('age' in keep_vars) & (len(self.years) > 1):
            keep_vars.remove('age')
            keep_vars.append('bene_dob')
            print(
                _mywrap(
                    """\
            Warning: Can't export age variable, exporting bene_dob instead
            """))

        tokeep_regex = []
        tokeep_regex.append(r'^(ehic)$')
        if gender is not None:
            tokeep_regex.append(r'^(sex)$')
        if ages is not None:
            tokeep_regex.append(r'^(age)$')
        if races is not None:
            tokeep_regex.append(r'^({})$'.format(race_col))
        if buyin_val is not None:
            tokeep_regex.append(r'^(buyin\d{2})$')
        if hmo_val is not None:
            tokeep_regex.append(r'^(hmoind\d{2})$')
        if self.year_type == 'age':
            tokeep_regex.append(r'^(bene_dob)$')
        if keep_vars is not None:
            for var in keep_vars:
                tokeep_regex.append(r'^({})$'.format(var))

        tokeep_regex = '|'.join(tokeep_regex)

        tokeep_vars = {}
        for year in self.years:
            if self.parquet_engine == 'pyarrow':
                pf = pq.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                cols = pf.schema.names
            elif self.parquet_engine == 'fastparquet':
                pf = fp.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                cols = pf.columns

            tokeep_vars[year] = [x for x in cols if re.search(tokeep_regex, x)]

            # Check cols against keep_vars
            # Is there an item in keep_vars that wasn't matched?
            for var in keep_vars:
                if [x for x in tokeep_vars[year] if re.search(var, x)] == []:
                    msg = f"""\
                    WARNING: variable `{var}` in the keep_vars argument
                    was not found in bsfab for year {year}
                    """
                    print(_mywrap(msg))

        # Now perform extraction
        extracted_dfs = []
        if not dask:
            nobs_dropped = {year: {} for year in self.years}

        # Do filtering for all vars that are
        # checkable within a single year's data
        for year in self.years:
            if verbose:
                msg = f"""\
                Importing bsfab file
                - year: {year}
                - columns: {tokeep_vars[year]}
                - time elapsed: {(time() - t0) / 60:.2f} minutes
                """
                print(_mywrap(msg))

            if dask:
                pl = dd.read_parquet(
                    self._fpath(self.percent, year, 'bsfab'),
                    columns=tokeep_vars[year],
                    index=['bene_id'],
                    engine=self.parquet_engine)
            elif self.parquet_engine == 'pyarrow':
                pf = pq.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                pl = pf.read(
                    columns=tokeep_vars[year],
                    nthreads=min(len(tokeep_vars[year]),
                                 self.parquet_nthreads)).to_pandas().set_index(
                                     'bene_id')
            elif self.parquet_engine == 'fastparquet':
                pf = fp.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                pl = pf.to_pandas(columns=tokeep_vars[year], index='bene_id')

            if not dask:
                nobs = len(pl)

            if gender is not None:
                if pl['sex'].dtype.name == 'category':
                    if pl['sex'].dtype.categories.dtype == object:
                        var_type = 'string'
                    else:
                        var_type = 'numeric'
                elif np.issubdtype(pl['sex'].dtype, np.number):
                    var_type = 'numeric'
                else:
                    var_type = 'string'

                if var_type == 'string':
                    pl = pl.loc[pl['sex'] == gender]
                else:
                    pl = pl.loc[pl['sex'] == int(gender)]

                if 'sex' not in keep_vars:
                    pl = pl.drop('sex', axis=1)

                if not dask:
                    nobs_dropped[year]['gender'] = 1 - (len(pl) / nobs)
                    nobs = len(pl)

            if ages is not None:
                pl = pl.loc[pl['age'].isin(ages)]

                if not dask:
                    nobs_dropped[year]['age'] = 1 - (len(pl) / nobs)
                    nobs = len(pl)

                if 'age' not in keep_vars:
                    pl = pl.drop('age', axis=1)

            if races is not None:
                pl = pl.loc[pl[race_col].isin(races)]

                if not dask:
                    nobs_dropped[year]['race'] = 1 - (len(pl) / nobs)
                    nobs = len(pl)

                if race_col not in keep_vars:
                    pl = pl.drop(race_col, axis=1)

            if (buyin_val is not None) and (self.year_type == 'calendar'):
                regex = re.compile(r'^buyin\d{2}').search
                buyin_cols = [x for x in pl.columns if regex(x)]
                pl = pl.loc[(pl[buyin_cols].isin(buyin_val)).all(axis=1)]
                pl = pl.drop(set(buyin_cols).difference(keep_vars), axis=1)

                if not dask:
                    nobs_dropped[year]['buyin_val'] = 1 - (len(pl) / nobs)
                    nobs = len(pl)

            if (hmo_val is not None) and (self.year_type == 'calendar'):
                regex = re.compile(r'^hmoind\d{2}').search
                hmo_cols = [x for x in pl.columns if regex(x)]
                pl = pl.loc[(pl[hmo_cols].isin(hmo_val)).all(axis=1)]
                pl = pl.drop(set(hmo_cols).difference(keep_vars), axis=1)

                if not dask:
                    nobs_dropped[year]['hmo_val'] = 1 - (len(pl) / nobs)
                    nobs = len(pl)

            pl.columns = [f'{x}{year}' for x in pl.columns]

            # Indicator for which patients exist in which year
            if (join != 'inner') or (self.year_type == 'age'):
                pl[f'match_{year}'] = True

            extracted_dfs.append(pl)

        if verbose & (len(extracted_dfs) > 1):
            msg = f"""\
            Merging together beneficiary files
            - years: {list(self.years)}
            - merge type: {join}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        # Unless no inter-year variables to check, always do an outer join.
        # Then after checking, perform desired join
        if len(extracted_dfs) == 1:
            pl = extracted_dfs[0]
        elif self.year_type == 'age':
            if len(extracted_dfs) == 2:
                pl = extracted_dfs[0].join(extracted_dfs[1], how='left')
            else:
                if not dask:
                    pl = extracted_dfs[0].join(
                        extracted_dfs[1:-1], how='outer').join(
                            extracted_dfs[-1], how='left')
                else:
                    pl = extracted_dfs[0]
                    for i in range(1, len(extracted_dfs) - 1):
                        pl = pl.join(extracted_dfs[i], how='outer')
                    pl = pl.join(extracted_dfs[-1], how='left')
        elif join == 'right':
            if not dask:
                pl = extracted_dfs[-1].join(extracted_dfs[:-1], how='left')
            else:
                pl = extracted_dfs[-1]
                for i in range(len(extracted_dfs) - 1):
                    pl = pl.join(extracted_dfs[i], how='left')
        else:
            if not dask:
                pl = extracted_dfs[0].join(extracted_dfs[1:], how=join)
            else:
                pl = extracted_dfs[0]
                for i in range(1, len(extracted_dfs)):
                    pl = pl.join(extracted_dfs[i], how='left')

        pl.index.name = 'bene_id'

        if (join != 'inner') or (self.year_type == 'age'):
            if self.year_type == 'age':
                # Don't need last calendar year match variable
                # Am only matching part of last calendar year when doing age-
                # year match
                cols = [f'match_{year}' for year in self.years]
                cols.remove(f'match_{max(self.years)}')
                pl[cols] = pl[cols].fillna(False)
                pl = pl.drop(f'match_{max(self.years)}', axis=1)
            else:
                cols = [f'match_{year}' for year in self.years]
                pl[cols] = pl[cols].fillna(False)

        if ((buyin_val is not None) or
            (hmo_val is not None)) and (self.year_type == 'age'):

            # Create month of birth variable
            pl['bene_dob'] = pd.NaT
            for year in self.years:
                pl['bene_dob'] = pl['bene_dob'].combine_first(
                    pl[f'bene_dob{year}'])
                pl = pl.drop(f'bene_dob{year}', axis=1)

            pl['dob_month'] = pl['bene_dob'].dt.month

        if (buyin_val is not None) and (self.year_type == 'age'):
            if verbose:
                msg = f"""\
                Filtering based on buyin_val
                - values: {buyin_val}
                - year_type: {self.year_type}
                - time elapsed: {(time() - t0) / 60:.2f} minutes
                """
                print(_mywrap(msg))

            # Create indicator variable for each year if `buyin ==
            # buyin_val` for the 13 months starting in birthday month of
            # `year` and ending in birthday month of `year + 1`

            for year in self.years[:-1]:
                if not dask:
                    nobs = len(pl[pl[f'match_{year}']])

                regex = re.compile(r'buyin(\d{2})(\d{4})').search
                for month in range(1, 13):
                    cols = [m for col in pl.columns for m in [regex(col)] if m]
                    cols = [
                        m[0]
                        for m in cols
                        if ((int(m[1]) >= month) & (int(m[2]) == year)) or (
                            (int(m[1]) <= month) & (int(m[2]) == year + 1))]

                    pl[f'match_{year}'] = pl[f'match_{year}'].mask(
                        (pl['dob_month'] == month) &
                        (~pl[cols].isin(buyin_val)).all(axis=1),
                        False)

                if not dask:
                    nobs_dropped[year]['buyin'] = (
                        1 - (pl[f'match_{year}'].sum() / nobs))

            regex = re.compile(r'^buyin\d{2}\d{4}$').search
            pl = pl.drop([x for x in pl.columns if regex(x)], axis=1)

        if (hmo_val is not None) and (self.year_type == 'age'):
            if verbose:
                msg = f"""\
                Filtering based on hmo_val
                - values: {hmo_val}
                - year_type: {self.year_type}
                - time elapsed: {(time() - t0) / 60:.2f} minutes
                """
                print(_mywrap(msg))

            # Create indicator variable for each year if `hmo ==
            # hmo_val` for the 13 months starting in birthday month of
            # `year` and ending in birthday month of `year + 1`

            for year in self.years[:-1]:
                if not dask:
                    nobs = len(pl[pl[f'match_{year}']])

                regex = re.compile(r'hmoind(\d{2})(\d{4})').search
                for month in range(1, 13):
                    cols = [m for col in pl.columns for m in [regex(col)] if m]
                    cols = [
                        m[0]
                        for m in cols
                        if ((int(m[1]) >= month) & (int(m[2]) == year)) or (
                            (int(m[1]) <= month) & (int(m[2]) == year + 1))]

                    pl[f'match_{year}'] = pl[f'match_{year}'].mask(
                        (pl['dob_month'] == month) &
                        (~pl[cols].isin(hmo_val)).all(axis=1),
                        False)

                if not dask:
                    nobs_dropped[year]['hmo'] = (
                        1 - (pl[f'match_{year}'].sum() / len(pl)))

            regex = re.compile(r'^hmoind\d{2}\d{4}$').search
            pl = pl.drop([x for x in pl.columns if regex(x)], axis=1)

        if ((buyin_val is not None) or
            (hmo_val is not None)) and (self.year_type == 'age'):

            to_drop = ['dob_month']
            if 'bene_dob' not in keep_vars:
                to_drop.append('bene_dob')

            pl = pl.drop(to_drop, axis=1)

        if self.year_type == 'age':
            # Do correct filtering of data based on desired join
            match_cols = [f'match_{year}' for year in self.years[:-1]]
            if join == 'inner':
                pl = pl.loc[pl[match_cols].all(axis=1)]
                pl = pl.drop(match_cols, axis=1)
            elif join == 'outer':
                pl = pl.loc[pl[match_cols].any(axis=1)]
            elif join == 'left':
                pl = pl.loc[pl[f'match_{min(self.years)}']]
            elif join == 'right':
                pl = pl.loc[pl[f'match_{max(self.years) - 1}']]

        # Create single variable across years for any non month-oriented vars
        # Columns that vary by year:
        regex = re.compile(r'(?!_).\d{4}$').search
        year_cols = [x for x in pl.columns if regex(x)]

        # unique names of columns that vary by year:
        year_cols_stub = list(set([x[:-4] for x in year_cols]))

        for col in year_cols_stub:
            dest_col = f'{col}{min(self.years)}'

            if len(self.years) == 1:
                pl = pl.rename(columns={dest_col: col})
                continue

            for year in self.years[1:]:
                try:
                    pl[dest_col] = pl[dest_col].combine_first(
                        pl[f'{col}{year}'])
                    pl = pl.drop(f'{col}{year}', axis=1)
                except KeyError:
                    # Means that f'{col}{year}' wasn't loaded
                    pass

            pl = pl.rename(columns={dest_col: col})

        if not dask:
            self.nobs_dropped = nobs_dropped
            self.pl = pl
        else:
            self.pl = pl.compute()

        if verbose:
            msg = f"""\
            Finished cohort retrieval
            - percent sample: {self.percent}
            - years: {list(self.years)}
            - ages: {list(ages) if ages else None}
            - races: {races if races else None}
            - buyin values: {buyin_val}
            - HMO values: {hmo_val}
            - extra variables: {keep_vars}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(_mywrap(msg))

    @staticmethod
    def _get_pattern(obj):
        """
        If str, returns str. If compiled regex, returns pattern
        """
        if type(obj) == str:
            return obj
        elif isinstance(obj, re._pattern_type):
            return obj.pattern
        else:
            raise TypeError('Provided non string or regex to get_pattern()')

    def _create_rename_dict(
            self, hcpcs=None, icd9_dx=None, icd9_sg=None, rename={}):
        """
        Make dictionary where the keys are codes/pattern strings and values are
        new column names
        """

        # If the values of rename are lists, make sure they match up on length
        msg = f"""\
        If the values of the rename dictionary are lists, they need
        to match the length of the list of codes provided
        """
        msg = _mywrap(msg)
        if type(rename.get('hcpcs')) == list:
            assert len(rename.get('hcpcs')) == len(hcpcs), msg
        if type(rename.get('icd9_dx')) == list:
            assert len(rename.get('icd9_dx')) == len(icd9_dx), msg
        if type(rename.get('icd9_sg')) == list:
            assert len(rename.get('icd9_sg')) == len(icd9_sg), msg

        # Generate a key with empty value for every variable
        rename_new = {}
        if hcpcs is not None:
            for item in hcpcs:
                rename_new[self._get_pattern(item)] = ''

        if icd9_dx is not None:
            for item in icd9_dx:
                rename_new[self._get_pattern(item)] = ''

        if icd9_sg is not None:
            for item in icd9_sg:
                rename_new[self._get_pattern(item)] = ''

        # Now fill in rename_new using rename
        msg = 'The values of the rename dictionary must be type list or dict'
        if type(rename.get('hcpcs')) == list:
            for i in range(len(rename['hcpcs'])):
                key = self._get_pattern(hcpcs[i])
                val = rename['hcpcs'][i]
                rename_new[key] = val
        elif type(rename.get('hcpcs')) == dict:
            rename_new = {**rename_new, **rename['hcpcs']}
        elif rename.get('hcpcs') == None:
            pass
        else:
            raise TypeError(msg)

        if type(rename.get('icd9_dx')) == list:
            for i in range(len(rename['icd9_dx'])):
                key = self._get_pattern(icd9_dx[i])
                val = rename['icd9_dx'][i]
                rename_new[key] = val
        elif type(rename.get('icd9_dx')) == dict:
            rename_new = {**rename_new, **rename['icd9_dx']}
        elif rename.get('icd9_dx') == None:
            pass
        else:
            raise TypeError(msg)

        if type(rename.get('icd9_sg')) == list:
            for i in range(len(rename['icd9_sg'])):
                key = self._get_pattern(icd9_sg[i])
                val = rename['icd9_sg'][i]
                rename_new[key] = val
        elif type(rename.get('icd9_sg')) == dict:
            rename_new = {**rename_new, **rename['icd9_sg']}
        elif rename.get('icd9_sg') == None:
            pass
        else:
            raise TypeError(msg)

        rename_new = {k: v for k, v in rename_new.items() if v != ''}
        return rename_new

    def _search_for_codes_type_check(
            self, data_types, hcpcs, icd9_dx, icd9_dx_max_cols, icd9_sg,
            keep_vars, collapse_codes, rename, convert_ehic, verbose):
        """Check types and valid values for :func:`search_for_codes`

        Also resolves input into correct value
        """

        if type(data_types) is str:
            data_types = [data_types]

        data_types = set(data_types)
        ok_data_types = ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']

        # Check that all data types provided to search through exist
        if not data_types.issubset(ok_data_types):
            invalid_vals = list(data_types.difference(ok_data_types))
            msg = f"""\
            {invalid_vals} does not match any dataset.
            Allowed `data_types` are {ok_data_types}.
            """
            raise ValueError(_mywrap(msg))

        # Assert that keep_vars is a dict and that the keys are in ok_data_types
        assert isinstance(keep_vars, dict)
        if not set(keep_vars.keys()).issubset(ok_data_types):
            invalid_vals = list(set(keep_vars.keys()).difference(ok_data_types))
            msg = f"""\
            {invalid_vals} does not match any dataset.
            Allowed keys of `keep_vars` are {ok_data_types}.
            """
            raise ValueError(_mywrap(msg))

        # Instantiate all data types in the keep_vars dict
        for data_type in ok_data_types:
            keep_vars[data_type] = keep_vars.get(data_type, [])

            if type(keep_vars[data_type]) is str:
                keep_vars[data_type] = [keep_vars[data_type]]

        codes = {'hcpcs': hcpcs, 'icd9_dx': icd9_dx, 'icd9_sg': icd9_sg}

        msg = f"""\
        Codes to search through must be str, compiled regex, or a list of
        either.
        """
        all_codes = []
        for name, code in codes.items():
            if code is None:
                continue
            if type(code) == str:
                code = [code]
            elif isinstance(code, re._pattern_type):
                code = [code]
            elif type(code) == list:
                # Check all elements of list are same type
                if type(code[0]) == str:
                    assert all((type(x) is str) for x in code)
                elif isinstance(code[0], re._pattern_type):
                    assert all(isinstance(x, re._pattern_type) for x in code)
                else:
                    raise TypeError(_mywrap(msg))
            else:
                raise TypeError(_mywrap(msg))

            codes[name] = code
            all_codes.extend(code)

        hcpcs = codes['hcpcs']
        icd9_dx = codes['icd9_dx']
        icd9_sg = codes['icd9_sg']

        assert type(collapse_codes) == bool
        assert isinstance(rename, dict)

        if not set(rename.keys()).issubset(['hcpcs', 'icd9_dx', 'icd9_sg']):
            msg = f"""\
            Allowed keys of `rename` are ['hcpcs', 'icd9_dx', 'icd9_sg'].
            """
            raise ValueError(_mywrap(msg))

        if not collapse_codes:
            all_codes = [self._get_pattern(x) for x in all_codes]
            msg = 'Code patterns given must be unique'
            assert len(all_codes) == len(set(all_codes)), msg

        if collapse_codes and any([x is not None for x in rename.values()]):
            msg = f"""\
            rename argument not allowed when collapse_codes is True
            """
            raise ValueError(_mywrap(msg))

        if (icd9_dx is None) and (icd9_dx_max_cols is not None):
            msg = f"""\
            icd9_dx_max_cols argument not allowed when icd9_dx is None
            """
            raise ValueError(_mywrap(msg))

        assert type(convert_ehic) == bool
        assert type(verbose) == bool

        Return = namedtuple(
            'variables', [
                'data_types', 'hcpcs', 'icd9_dx', 'icd9_dx_max_cols', 'icd9_sg',
                'keep_vars', 'collapse_codes', 'rename', 'convert_ehic',
                'verbose'])
        return Return(
            data_types=data_types,
            hcpcs=hcpcs,
            icd9_dx=icd9_dx,
            icd9_dx_max_cols=icd9_dx_max_cols,
            icd9_sg=icd9_sg,
            keep_vars=keep_vars,
            collapse_codes=collapse_codes,
            rename=rename,
            convert_ehic=convert_ehic,
            verbose=verbose)

    def search_for_codes(
            self,
            data_types: Union[str, List[str]],
            hcpcs: Union[str, Pattern, List[str], List[Pattern], None] = None,
            icd9_dx: Union[str, Pattern, List[str], List[Pattern], None] = None,
            icd9_dx_max_cols: Optional[int] = None,
            icd9_sg: Union[str, Pattern, List[str], List[Pattern], None] = None,
            keep_vars: Dict[str, Union[str, List[str]]] = {},
            collapse_codes: bool = True,
            rename: Dict[str, Union[str, List[str], Dict[str, str], None]] = {
                'hcpcs': None,
                'icd9_dx': None,
                'icd9_sg': None},
            convert_ehic: bool = True,
            verbose=False):
        """Search in claim-level datasets for HCPCS and/or ICD9 codes

        Note: Each code given must be distinct, or ``collapse_codes`` must be ``True``

        Args:
            data_types:
                Files to search through. The following are allowed:

                - ``carc``  (`Carrier File, Claims segment`_)
                - ``carl``  (`Carrier File, Line segment`_)
                - ``ipc``   (`Inpatient File, Claims segment`_)
                - ``ipr``   (`Inpatient File, Revenue Center segment`_)
                - ``med``   (`MedPAR File`_)
                - ``opc``   (`Outpatient File, Claims segment`_)
                - ``opr``   (`Outpatient File, Revenue Center segment`_)

                .. _`Carrier File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/carrier-rif/#carrier-rif_1
                .. _`Carrier File, Line segment`: https://kylebarron.github.io/medicare-documentation/resdac/carrier-rif/#line-file
                .. _`Inpatient File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/ip-rif/#inpatient-rif_1
                .. _`Inpatient File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/ip-rif/#revenue-center-file
                .. _`MedPAR File`: https://kylebarron.github.io/medicare-documentation/resdac/medpar-rif/#medpar-rif_1
                .. _`Outpatient File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/op-rif/#outpatient-rif_1
                .. _`Outpatient File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/op-rif/#revenue-center-file
            hcpcs: HCPCS codes to search for
            icd9_dx: ICD-9 diagnosis codes to search for
            icd9_dx_max_cols: Max number of ICD9 diagnosis code columns to
                search through. If ``None``, will search through all columns.
            icd9_sg: ICD-9 procedure codes to search for
            keep_vars: column names to return
            collapse_codes: If ``True``, returns a single column named
                ``match`` that is ``True`` for the claims with a matched code and ``False`` otherwise. If ``collapse_codes`` is ``False``, returns a column for each code provided
            rename: Match columns to rename when ``collapse_codes`` is ``False``.
            convert_ehic: If ``True``, merges on ``bene_id`` for years <
                2006
            verbose: Print progress of program to console

        Returns:
            Creates ``.cl`` attribute. This is a dict where keys are the
            data_types provided and values are pandas DataFrames with
            ``bene_id`` as index and indicator columns for each code provided.
        """

        if self.verbose or verbose:
            verbose = True
            t0 = time()

        objs = self._search_for_codes_type_check(
            data_types=data_types,
            hcpcs=hcpcs,
            icd9_dx=icd9_dx,
            icd9_dx_max_cols=icd9_dx_max_cols,
            icd9_sg=icd9_sg,
            keep_vars=keep_vars,
            collapse_codes=collapse_codes,
            rename=rename,
            convert_ehic=convert_ehic,
            verbose=verbose)
        data_types = objs.data_types
        hcpcs = objs.hcpcs
        icd9_dx = objs.icd9_dx
        icd9_dx_max_cols = objs.icd9_dx_max_cols
        icd9_sg = objs.icd9_sg
        keep_vars = objs.keep_vars
        collapse_codes = objs.collapse_codes
        rename = objs.rename
        convert_ehic = objs.convert_ehic
        verbose = objs.verbose

        ok_hcpcs_data_types = ['carl', 'ipr', 'opr']
        ok_dx_data_types = ['carc', 'carl', 'ipc', 'med', 'opc']
        ok_sg_data_types = ['ipc', 'med', 'opc']

        # Print which codes are searched in which dataset
        if verbose and ((hcpcs or icd9_dx or icd9_sg) is not None):
            msg = f"""\
            Will check the following codes
            - percent sample: {pct_dict[self.percent]}%
            - years: {list(self.years)}
            """
            msg = _mywrap(msg)

            if hcpcs is not None:
                dts = list(data_types.intersection(ok_hcpcs_data_types))
                if dts != []:
                    msg += _mywrap(
                        f"""\
                    - HCPCS codes: {hcpcs}
                      in data types: {dts}
                    """)

            if icd9_dx is not None:
                dts = list(data_types.intersection(ok_dx_data_types))
                if dts != []:
                    msg += _mywrap(
                        f"""\
                    - ICD-9 diagnosis codes: {icd9_dx}
                      in data types: {dts}
                    """)

            if icd9_sg is not None:
                dts = list(data_types.intersection(ok_sg_data_types))
                if dts != []:
                    msg += _mywrap(
                        f"""\
                    - ICD-9 procedure codes: {icd9_sg}
                      in data types: {dts}
                    """)

            print(msg)

        if not all([x is None for x in rename.values()]):
            rename = self._create_rename_dict(
                hcpcs=hcpcs, icd9_dx=icd9_dx, icd9_sg=icd9_sg, rename=rename)

        data = {}
        for data_type in data_types:
            data[data_type] = {}
            for year in self.years:
                if verbose:
                    msg = _mywrap(
                        f"""\
                    Starting search for codes
                    - year: {year}
                    - data type: {data_type}
                    """)
                    if data_type in ok_hcpcs_data_types:
                        if hcpcs is not None:
                            msg += _mywrap(
                                f"""\
                            - HCPCS codes: {hcpcs}
                            """)
                    if data_type in ok_dx_data_types:
                        if icd9_dx is not None:
                            msg += _mywrap(
                                f"""\
                            - ICD-9 diagnosis codes: {icd9_dx}
                            """)
                    if data_type in ok_sg_data_types:
                        if icd9_sg is not None:
                            msg += _mywrap(
                                f"""\
                            - ICD-9 procedure codes: {icd9_sg}
                            """)
                    if keep_vars[data_type] != []:
                        msg += _mywrap(
                            f"""\
                        - Keeping variables: {keep_vars[data_type]}
                        """)
                    msg += _mywrap(
                        f"""\
                    - time elapsed: {(time() - t0) / 60:.2f} minutes
                    """)
                    print(msg)

                data[data_type][year] = self._search_for_codes_single_year(
                    year=year,
                    data_type=data_type,
                    hcpcs=(hcpcs if data_type in ok_hcpcs_data_types else None),
                    icd9_dx=(
                        icd9_dx if data_type in ok_dx_data_types else None),
                    icd9_dx_max_cols=icd9_dx_max_cols,
                    icd9_sg=(
                        icd9_sg if data_type in ok_sg_data_types else None),
                    keep_vars=keep_vars[data_type],
                    collapse_codes=collapse_codes,
                    rename=rename)

        if verbose:
            msg = f"""\
            Concatenating matched codes across years
            - years: {list(self.years)}
            - data types: {data_types}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        years_ehic = [x for x in self.years if x < 2006]
        years_bene_id = [x for x in self.years if x >= 2006]

        # Don't go through ehic process if data is only post 2006
        if years_ehic == []:
            for data_type in data_types:
                data[data_type]['all'] = pd.concat([
                    data[data_type][year] for year in years_bene_id])
                for year in years_ehic:
                    data[data_type][year] = None
                data[data_type] = data[data_type]['all']

            self.cl = data
            return

        if (min(self.years) < 2006) and (max(self.years) >= 2006):
            convert_ehic = True

        # Concatenate ehic data (2005 and earlier)
        if (convert_ehic) and (min(self.years) < 2006):

            # If self.pl exists, then cl data frames use only those ids
            # So I can merge using that
            if self.pl is not None:
                for data_type in data_types:
                    df = pd.concat([
                        data[data_type][year] for year in years_ehic])
                    df = df.merge(
                        self.pl, how='left', left_index=True, right_on='ehic')

                    data[data_type]['ehic'] = df

            else:
                for year in years_ehic:
                    # Read in all bsfab data
                    if self.parquet_engine == 'pyarrow':
                        pf = pq.ParquetFile(
                            self._fpath(self.percent, year, 'bsfab'))
                        pl = pf.read(
                            columns=['ehic', 'bene_id'],
                            nthreads=2).to_pandas().set_index('ehic')
                    elif self.parquet_engine == 'fastparquet':
                        pf = fp.ParquetFile(
                            self._fpath(self.percent, year, 'bsfab'))
                        pl = pf.to_pandas(columns=['bene_id'], index='ehic')

                    # Join bene_ids onto data using ehic
                    for data_type in data_types:
                        data[data_type][year] = data[data_type][year].join(
                            pl, how='left').reset_index().set_index('bene_id')

                for data_type in data_types:
                    data[data_type]['ehic'] = pd.concat([
                        data[data_type][year] for year in years_ehic])

        elif (not convert_ehic) and (min(self.years) < 2006):
            for data_type in data_types:
                data[data_type]['ehic'] = pd.concat([
                    data[data_type][year] for year in years_ehic])

        for data_type in data_types:
            # Delete single-year ehic data
            for year in years_ehic:
                data[data_type][year] = None

            # Concatenate bene_id data (2006 and later)
            data[data_type]['bene_id'] = pd.concat([
                data[data_type][year] for year in years_bene_id])

            # Delete single-year bene_id data
            for year in years_bene_id:
                data[data_type][year] = None

            # Concatenate ehic data with bene_id data
            if data[data_type]['ehic'].index.name == data[data_type][
                    'bene_id'].index.name:
                data[data_type]['all'] = pd.concat([
                    data[data_type]['ehic'], data[data_type]['bene_id']])

                data[data_type]['ehic'] = None
                data[data_type]['bene_id'] = None

            else:
                data[data_type]['all'] = pd.concat([
                    data[data_type]['ehic'].reset_index(),
                    data[data_type]['bene_id'].reset_index()],
                                                   ignore_index=True)

                data[data_type]['ehic'] = None
                data[data_type]['bene_id'] = None

            data[data_type] = data[data_type]['all']

        self.cl = data

        if verbose:
            msg = f"""\
            Finished searching for codes
            - percent sample: {self.percent}
            - years: {list(self.years)}
            - data_types: {data_types}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(_mywrap(msg))

    def _search_for_codes_single_year(
            self,
            year,
            data_type,
            hcpcs=None,
            icd9_dx=None,
            icd9_dx_max_cols=None,
            icd9_sg=None,
            keep_vars=[],
            rename={},
            collapse_codes=False):
        """Search in a single claim-level dataset for HCPCS/ICD9 codes

        Note: Each code given must be distinct, or collapse_codes must be True

        Args:
            year (int): year of data to search
            data_type (str): One of carc, carl, ipc, ipr, med, opc, opr
            hcpcs (str, compiled regex, list[str], list[compiled regex]):
                List of HCPCS codes to look for
            icd9_dx (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 diagnosis codes to look for
            icd9_dx_max_cols (int): Max number of ICD9 diagnosis code columns to
                search through
            icd9_sg (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 procedure codes to look for
            keep_vars (list[str]): list of column names to return
            rename (dict): dictionary where keys are codes to match, and values
                are new column names
            collapse_codes (bool): If True, returns a single column "match";
                else it returns a column for each code provided

        Returns:
            DataFrame with bene_id and bool columns for each code to search for
        """

        is_search_for_codes = (hcpcs or icd9_dx or icd9_sg) is not None

        if year < 2006:
            pl_id_col = 'ehic'
        else:
            pl_id_col = 'bene_id'

        # Assumes bene_id or ehic is index name or name of a column
        if self.pl is not None:
            if pl_id_col == self.pl.index.name:
                pl_ids_to_filter = self.pl.index
            else:
                pl_ids_to_filter = self.pl[pl_id_col].values
        else:
            pl_ids_to_filter = None

        # Determine which variables to extract
        regex_string = []
        if data_type == 'med':
            cl_id_regex = r'^medparid$'
            regex_string.append(cl_id_regex)
        else:
            cl_id_regex = r'^clm_id$|^claimindex$'
            regex_string.append(cl_id_regex)

        regex_string.append(r'^bene_id$')
        regex_string.append(r'^ehic$')

        if hcpcs is not None:
            hcpcs_regex = r'^hcpcs_cd$'
            regex_string.append(hcpcs_regex)

        if icd9_dx is not None:
            if data_type == 'carl':
                icd9_dx_regex = r'icd_dgns_cd(\d*)$'
            elif data_type == 'med':
                icd9_dx_regex = r'^dgnscd(\d+)$$'
            else:
                icd9_dx_regex = r'^icd_dgns_cd(\d+)$'
            regex_string.append(icd9_dx_regex)

        if icd9_sg is not None:
            icd9_sg_regex = r'^icd_prcdr_cd\d+$'
            regex_string.append(icd9_sg_regex)

        for var in keep_vars:
            regex_string.append(r'^{}$'.format(var))

        regex_string = '|'.join(regex_string)
        regex = re.compile(regex_string).search

        if self.parquet_engine == 'pyarrow':
            all_cols = pq.ParquetFile(
                self._fpath(self.percent, year, data_type)).schema.names
        elif self.parquet_engine == 'fastparquet':
            all_cols = fp.ParquetFile(
                self._fpath(self.percent, year, data_type)).columns
        cols = [x for x in all_cols if regex(x)]

        # Check cols against keep_vars
        # Is there an item in keep_vars that wasn't matched?
        for var in keep_vars:
            if [x for x in cols if re.search(var, x)] == []:
                msg = f"""\
                WARNING: variable `{var}` in the keep_vars argument
                was not found in {data_type}
                """
                print(_mywrap(msg))

        cl_id_col = [x for x in cols if re.search(cl_id_regex, x)]
        if hcpcs is not None:
            hcpcs_cols = [x for x in cols if re.search(hcpcs_regex, x)]
        else:
            hcpcs_cols = None

        if icd9_dx is not None:
            icd9_dx_cols = [x for x in cols if re.search(icd9_dx_regex, x)]

            if icd9_dx_max_cols is not None:
                new_cols = [
                    x for x in icd9_dx_cols
                    if int(re.search(icd9_dx_regex, x)[1]) <= icd9_dx_max_cols]

                deleted_cols = list(set(icd9_dx_cols).difference(new_cols))
                cols = [x for x in cols if x not in deleted_cols]
                icd9_dx_cols = new_cols
        else:
            icd9_dx_cols = None

        if icd9_sg is not None:
            icd9_sg_cols = [x for x in cols if re.search(icd9_sg_regex, x)]
        else:
            icd9_sg_cols = None

        # This holds the df's from each iteration over the claim-level dataset
        all_cl = []

        if self.parquet_engine == 'pyarrow':
            pf = pq.ParquetFile(self._fpath(self.percent, year, data_type))
            itr = (
                pf.read_row_group(
                    i,
                    columns=cols,
                    nthreads=min(len(cols), self.parquet_nthreads)).to_pandas()
                .set_index(pl_id_col) for i in range(pf.num_row_groups))
        elif self.parquet_engine == 'fastparquet':
            pf = fp.ParquetFile(self._fpath(self.percent, year, data_type))
            itr = pf.iter_row_groups(columns=cols, index=pl_id_col)

        for cl in itr:
            if pl_ids_to_filter is not None:
                index_name = cl.index.name
                cl = cl.join(pd.DataFrame(index=pl_ids_to_filter), how='inner')
                cl.index.name = index_name

            if not is_search_for_codes:
                all_cl.append(cl)
                continue

            # The index needs to be unique for the stuff I do below with first
            # saving all indices in a var idx, then using that with cl.loc[].
            # If index is bene_id, it'll set matched to true for _anyone_ who
            # had a match _sometime_.
            cl = cl.reset_index().set_index(cl_id_col)

            if collapse_codes:
                cl['match'] = False

                if hcpcs:
                    for code in hcpcs:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[hcpcs_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[hcpcs_cols] == code
                                   ).any(axis=1), 'match'] = True

                    cl = cl.drop(hcpcs_cols, axis=1)

                if icd9_dx:
                    for code in icd9_dx:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_dx_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_dx_cols] == code
                                   ).any(axis=1), 'match'] = True

                    cl = cl.drop(icd9_dx_cols, axis=1)

                if icd9_sg:
                    for code in icd9_sg:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_sg_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_sg_cols] == code
                                   ).any(axis=1), 'match'] = True

                    cl = cl.drop(icd9_sg_cols, axis=1)

                # Keep all rows; not just matches
                cl = cl.reset_index().set_index(pl_id_col)
                all_cl.append(cl)

            else:
                all_created_cols = []
                if hcpcs:
                    for code in hcpcs:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[hcpcs_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True
                            all_created_cols.append(code.pattern)

                        else:
                            cl[code] = False
                            idx = cl.index[(cl[hcpcs_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True
                            all_created_cols.append(code)

                    cl = cl.drop(hcpcs_cols, axis=1)

                if icd9_dx:
                    for code in icd9_dx:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[icd9_dx_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True
                            all_created_cols.append(code.pattern)

                        else:
                            cl[code] = False
                            idx = cl.index[(
                                cl[icd9_dx_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True
                            all_created_cols.append(code)

                    cl = cl.drop(icd9_dx_cols, axis=1)

                if icd9_sg:
                    for code in icd9_sg:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[icd9_sg_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True
                            all_created_cols.append(code.pattern)

                        else:
                            cl[code] = False
                            idx = cl.index[(
                                cl[icd9_sg_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True
                            all_created_cols.append(code)

                    cl = cl.drop(icd9_sg_cols, axis=1)

                cl['match'] = (cl[all_created_cols] == True).any(axis=1)

                # Rename columns according to `rename` dictionary
                cl = cl.rename(columns=rename)

                cl = cl.reset_index().set_index(pl_id_col)
                all_cl.append(cl)

        cl = pd.concat(all_cl, axis=0)
        cl['year'] = np.uint16(year)

        # Merge back onto pl_ids_to_filter so that claim-level df
        # has same index values as person-level df
        # cl = cl.join(
        #     pd.DataFrame(index=pl_ids_to_filter),
        #     how='outer')

        # Revert to the following if change index back to cl_id_col
        # cl = cl.reset_index().merge(
        #     pd.DataFrame(index=pl_ids_to_filter),
        #     how='outer',
        #     left_on=pl_id_col,
        #     right_index=True).set_index(pl_id_col)

        return cl

    def _search_for_codes_pl(
            self,
            data_types,
            hcpcs=None,
            icd9_dx=None,
            icd9_sg=None,
            collapse_codes=False):

        cl = self.cl
        if self.pl is not None:
            pl = self.pl

        if collapse_codes:
            bene_id_idx = cl.index[cl['match'] == True]  # noqa

            if 'match' not in pl.columns:
                pl['match'] = False

            pl.loc[bene_id_idx, 'match'] = True

        else:
            if hcpcs:
                for code in hcpcs:
                    if isinstance(code, re._pattern_type):
                        if code.pattern not in pl.columns:
                            pl[code.pattern] = False
                        idx = cl.index[cl[code.pattern] == True]  # noqa
                        pl.loc[idx, code.pattern] = True

                    else:
                        if code not in pl.columns:
                            pl[code] = False
                        idx = cl.index[cl[code] == True]  # noqa
                        pl.loc[idx, code] = True

            if icd9_dx:
                for code in icd9_dx:
                    if isinstance(code, re._pattern_type):
                        if code.pattern not in pl.columns:
                            pl[code.pattern] = False
                        idx = cl.index[cl[code.pattern] == True]  # noqa
                        pl.loc[idx, code.pattern] = True

                    else:
                        if code not in pl.columns:
                            pl[code] = False
                        idx = cl.index[cl[code] == True]  # noqa
                        pl.loc[idx, code] = True

            if icd9_sg:
                for code in icd9_sg:
                    if isinstance(code, re._pattern_type):
                        if code.pattern not in pl.columns:
                            pl[code.pattern] = False
                        idx = cl.index[cl[code.pattern] == True]  # noqa
                        pl.loc[idx, code.pattern] = True

                    else:
                        if code not in pl.columns:
                            pl[code] = False
                        idx = cl.index[cl[code] == True]  # noqa
                        pl.loc[idx, code] = True

        return pl
