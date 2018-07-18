#! /usr/bin/env python3
import re
import numpy as np
import warnings
# Note, take this out when this is resolved
# https://github.com/pandas-dev/pandas/issues/21952
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import fastparquet as fp
import dask.dataframe as dd
import pyarrow.parquet as pq
# TODO split fastparquet and pyarrow into conditional imports

from time import time
from typing import Dict, List, NamedTuple, Optional, Pattern, Union
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

    Args:
        percent: percent sample of data to use
        years: years of data to use
        year_type: ``calendar`` to work with multiple years as calendar years;
            ``age`` to work with patients' age years
        dask: Use dask library for out of core computation
        verbose: Print progress status of program to console
        parquet_engine: ``pyarrow`` or ``fastparquet``
        parquet_nthreads: number of threads to use when reading file
        dta_path: path to Stata Medicare files
        pq_path: path to Parquet Medicare files
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
            dask: bool = False,
            verbose: bool = False,
            parquet_engine: str = 'pyarrow',
            parquet_nthreads: Optional[int] = None,
            dta_path: str = '/disk/aging/medicare/data',
            pq_path:
            str = '/disk/agebulk3/medicare.work/doyle-dua51929/barronk-dua51929/raw/pq'
            ) -> None: # yapf: disable
        """Instantiate a MedicareDF object"""

        # Check types
        if type(percent) in (float, int):
            try:
                self.percent = pct_dict[percent]
            except KeyError:
                msg = f"""\
                percent provided is not valid.
                Valid arguments are: {list(pct_dict.keys())}
                """
                raise ValueError(_mywrap(msg))
        elif isinstance(percent, str):
            if percent not in allowed_pcts:
                msg = f'percent must be one of: {allowed_pcts}'
                raise ValueError(msg)

            self.percent = percent
        else:
            raise TypeError('percent must be str or number')

        if isinstance(years, bool):
            raise TypeError('years must be a number or list of numbers')
        if type(years) == int:
            years = [years]
        elif isinstance(years, (list, range)):
            years = years
        else:
            raise TypeError('years must be a number or list of numbers')

        if (len(years) == 1) & (year_type == 'age'):
            msg = "year_type can't be `age` when only one year is given"
            raise ValueError(msg)

        self.years = years
        self.year_type = year_type
        self.verbose = verbose
        self.dask = dask
        self.tc = time()

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

    class _ReturnGetCohortTypeCheck(NamedTuple):
        gender: Optional[str]
        ages: Optional[List[int]]
        races: Optional[List[str]]
        rti_race: bool
        race_col: str
        buyin_val: Optional[List[str]]
        hmo_val: Optional[List[str]]
        join: str
        keep_vars: List[Union[str, Pattern]]
        dask: bool
        verbose: bool

    def _get_cohort_type_check(
            self,
            gender: Optional[str],
            ages: Union[int, List[int], None],
            races: Union[str, List[str], None],
            rti_race: bool,
            buyin_val: Union[str, List[str], None],
            hmo_val: Union[str, List[str], None],
            join: str,
            keep_vars: Union[str, Pattern, List[Union[str, Pattern]], None],
            dask: bool,
            verbose: bool) -> _ReturnGetCohortTypeCheck: # yapf: disable
        """Check types and valid values for :func:`get_cohort`

        Also resolves input into correct value
        """

        # Check gender
        if gender is None:
            pass
        elif isinstance(gender, str):
            try:
                gender = str(int(gender))
                if int(gender) not in range(0, 3):
                    raise ValueError(f'{gender} is invalid value for `gender`')
            except (ValueError, AssertionError):
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
        else:
            raise TypeError('gender must be str or None')

        # Check ages
        if (ages is None) | isinstance(ages, range):
            pass
        elif isinstance(ages, list):
            if not all(type(x) == int for x in ages):
                raise TypeError('ages must be int or list of ints')
        elif type(ages) == int:
            ages = [ages]
        else:
            raise TypeError('ages must be int or list of int')

        # check races
        if not isinstance(rti_race, bool):
            raise TypeError('rti_race must be bool')
        race_col = 'rti_race_cd' if rti_race else 'race'

        race_cbk = codebook('bsfab')[race_col]['values']
        race_cbk = {v.lower(): k for k, v in race_cbk.items()}
        if race_col == 'rti_race_cd':
            # Don't want races='hispanic' to match 'non-hispanic white'
            race_cbk['white'] = race_cbk.pop('non-hispanic white')

        if races is None:
            pass
        elif isinstance(races, list):
            try:
                races = [str(int(x)) for x in races]
                if any(int(x) not in range(0, 7) for x in races):
                    raise ValueError(f'{races} is invalid value for `races`')
            except ValueError:
                races_new = []
                for race in races:
                    r = [v for k, v in race_cbk.items() if race.lower() in k]
                    msg = f'`{race}` matches more than one race description'
                    assert len(r) <= 1, msg
                    msg = f'`{race}` matches no race description'
                    assert len(r) >= 1, msg
                    races_new.extend(r)

                races = races_new
        elif isinstance(races, str):
            try:
                races = [str(int(races))]
                if int(races[0]) not in range(0, 7):
                    raise ValueError(f'{races} is invalid value for `races`')
            except ValueError:
                r = [v for k, v in race_cbk.items() if races.lower() in k]
                msg = f'`{races}` matches more than one race description'
                assert len(r) <= 1, msg
                msg = f'`{races}` matches no race description'
                assert len(r) >= 1, msg

                races = r
        else:
            raise TypeError('races must be str or list of str')

        buyin_val = [buyin_val] if isinstance(buyin_val, str) else buyin_val
        hmo_val = [hmo_val] if isinstance(hmo_val, str) else hmo_val

        allowed_join = ['left', 'right', 'inner', 'outer']
        if join not in allowed_join:
            msg = f'join must be one of: {allowed_join}'
            raise ValueError(msg)

        msg = f"""\
        keep_vars must be str, compiled regex, or List[str, compiled regex]
        """
        if keep_vars is None:
            keep_vars = []
        if isinstance(keep_vars, (str, re._pattern_type)):
            keep_vars = [keep_vars]
        elif isinstance(keep_vars, list):
            if not all(isinstance(x, (str, re._pattern_type))
                       for x in keep_vars):
                raise TypeError(_mywrap(msg))
        else:
            raise TypeError(_mywrap(msg))

        if not isinstance(dask, bool):
            raise TypeError('dask must be type bool')
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be type bool')

        return self._ReturnGetCohortTypeCheck(
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

    def _get_cohort_get_vars_toload(
            self,
            gender: Optional[str],
            ages: Optional[List[int]],
            races: Optional[List[str]],
            race_col: str,
            buyin_val: Optional[List[str]],
            hmo_val: Optional[List[str]],
            keep_vars: List[Union[str, Pattern]]
            ) -> Dict[int, List[str]]: # yapf: disable
        """Get variables to import for each year

        Args:
            keep_vars: User-defined variables to keep in final dataset extract
            race_col: Name of race column used

        Returns:
            Names of variables to be loaded in each year
        """

        # Get list of variables to import for each year
        if self._str_in_keep_vars('age', keep_vars) & (len(self.years) > 1):
            keep_vars.remove('age')
            keep_vars.append('bene_dob')
            print("Warning: Can't export age, exporting bene_dob instead")

        toload_regex = []
        toload_regex.append(r'^(ehic)$')
        toload_regex.append(r'^(bene_id)$')
        if gender is not None:
            toload_regex.append(r'^(sex)$')
        if ages is not None:
            toload_regex.append(r'^(age)$')
        if races is not None:
            toload_regex.append(r'^({})$'.format(race_col))
        if buyin_val is not None:
            toload_regex.append(r'^(buyin\d{2})$')
        if hmo_val is not None:
            toload_regex.append(r'^(hmoind\d{2})$')
        if self.year_type == 'age':
            toload_regex.append(r'^(bene_dob)$')
        for keep_var in keep_vars:
            if isinstance(keep_var, str):
                toload_regex.append(r'^({})$'.format(keep_var))

        toload_regex = re.compile('|'.join(toload_regex)).search

        toload_vars: Dict[int, List[str]] = {}
        for year in self.years:
            if self.parquet_engine == 'pyarrow':
                pf = pq.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                cols = pf.schema.names
            elif self.parquet_engine == 'fastparquet':
                pf = fp.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
                cols = pf.columns

            toload_vars[year] = set(x for x in cols if toload_regex(x))
            for keep_var in keep_vars:
                if isinstance(keep_var, re._pattern_type):
                    toload_vars[year].update(
                        set(x for x in cols if keep_var.search(x)))

            # Check cols against keep_vars
            # Is there an item in keep_vars that wasn't matched?
            # NOTE need to check this against regex values of keep_vars
            for var in keep_vars:
                if [x for x in toload_vars[year] if re.search(var, x)] == []:
                    msg = f"""\
                    WARNING: variable `{var}` in the keep_vars argument
                    was not found in bsfab for year {year}
                    """
                    print(_mywrap(msg))

        return toload_vars

    def _get_cohort_extract_each_year(
            # yapf: disable
            self,
            year: int,
            toload_vars: List[str],
            nobs_dropped,
            gender: Optional[str],
            ages: Optional[List[int]],
            races: Optional[List[str]],
            rti_race: bool,
            race_col: str,
            buyin_val: Optional[List[str]],
            hmo_val: Optional[List[str]],
            join: str,
            keep_vars: List[str],
            dask: bool,
            verbose: bool
    ) -> (Union[pd.DataFrame, dd.DataFrame], Dict[int, Dict[str, float]]):
        # yapf: enable

        if verbose:
            msg = f"""\
            Importing bsfab file
            - year: {year}
            - columns: {toload_vars}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        if dask:
            pl = dd.read_parquet(
                self._fpath(self.percent, year, 'bsfab'),
                columns=toload_vars,
                index=['bene_id'],
                engine=self.parquet_engine)
        elif self.parquet_engine == 'pyarrow':
            pf = pq.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
            pl = pf.read(
                columns=toload_vars,
                nthreads=min(
                    len(toload_vars),
                    self.parquet_nthreads)).to_pandas().set_index('bene_id')
        elif self.parquet_engine == 'fastparquet':
            pf = fp.ParquetFile(self._fpath(self.percent, year, 'bsfab'))
            pl = pf.to_pandas(columns=toload_vars, index='bene_id')

        if not dask:
            nobs = len(pl)

        iemsg = 'Internal error: Missing column: '
        iemsg2 = '\nPlease submit a bug report at\n'
        iemsg2 += 'https://github.com/kylebarron/medicare_utils/issues/new'
        if gender is not None:
            assert 'sex' in pl.columns, iemsg + 'sex' + iemsg2
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

            if not self._str_in_keep_vars('sex', keep_vars):
                pl = pl.drop('sex', axis=1)

            if not dask:
                nobs_dropped[year]['gender'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

        if ages is not None:
            assert 'age' in pl.columns, iemsg + 'age' + iemsg2
            pl = pl.loc[pl['age'].isin(ages)]

            if not dask:
                nobs_dropped[year]['age'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            if not self._str_in_keep_vars('age', keep_vars):
                pl = pl.drop('age', axis=1)

        if races is not None:
            assert race_col in pl.columns, iemsg + race_col + iemsg2
            pl = pl.loc[pl[race_col].isin(races)]

            if not dask:
                nobs_dropped[year]['race'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            if not self._str_in_keep_vars(race_col, keep_vars):
                pl = pl.drop(race_col, axis=1)

        if (buyin_val is not None) and (self.year_type == 'calendar'):
            regex = re.compile(r'^buyin\d{2}').search
            buyin_cols = [x for x in pl.columns if regex(x)]
            msg = f'{iemsg}buyinXX{iemsg2}\n#cols: {len(buyin_cols)}'
            assert len(buyin_cols) == 12, msg

            pl = pl.loc[(pl[buyin_cols].isin(buyin_val)).all(axis=1)]
            pl = pl.drop(set(buyin_cols).difference(keep_vars), axis=1)

            if not dask:
                nobs_dropped[year]['buyin_val'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

        if (hmo_val is not None) and (self.year_type == 'calendar'):
            regex = re.compile(r'^hmoind\d{2}').search
            hmo_cols = [x for x in pl.columns if regex(x)]
            msg = f'{iemsg}hmoindXX{iemsg2}\n#cols: {len(hmo_cols)}'
            assert len(hmo_cols) == 12, msg

            pl = pl.loc[(pl[hmo_cols].isin(hmo_val)).all(axis=1)]
            pl = pl.drop(set(hmo_cols).difference(keep_vars), axis=1)

            if not dask:
                nobs_dropped[year]['hmo_val'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

        pl.columns = [f'{x}{year}' for x in pl.columns]

        # Indicator for which patients exist in which year
        if (join != 'inner') or (self.year_type == 'age'):
            pl[f'match_{year}'] = True

        return pl, nobs_dropped

    def _get_cohort_age_year_filter(
            # yapf: disable
            self,
            pl: Union[pd.DataFrame, dd.DataFrame],
            var: str,
            values,
            nobs_dropped: dict,
            dask: bool,
            verbose: bool
    ) -> (Union[pd.DataFrame, dd.DataFrame], Dict[int, Dict[str, float]]):
        # yapf: enable
        """Perform filtering for age-year-specific variables

        Args:
            pl: patient-level data
            var: variable to filter on. Either `buyin` or `hmoind`
            values: values of either `buyin_val` or `hmo_val`
            nobs_dropped: dict with nobs dropped in each year
            dask: whether to use dask
            verbose: log warnings to stdout

        Returns:
            (pd.DataFrame): patient-level data
            (dict): number of observations dropped with each restriction
        """

        assert var in ['buyin', 'hmoind']

        if verbose:
            msg = f"""\
            Filtering based on {var}_val
            - values: {values}
            - year_type: {self.year_type}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        # Create indicator variable for each year if `buyin ==
        # buyin_val` for the 13 months starting in birthday month of
        # `year` and ending in birthday month of `year + 1`

        for year in self.years[:-1]:
            if not dask:
                nobs = len(pl[pl[f'match_{year}']])

            regex_string = r'{}'.format(var) + r'(\d{2})(\d{4})'
            regex = re.compile(regex_string).search
            for month in range(1, 13):
                cols = [m for col in pl.columns for m in [regex(col)] if m]
                cols = [
                    m[0]
                    for m in cols
                    if ((int(m[1]) >= month) & (int(m[2]) == year)) or (
                        (int(m[1]) <= month) & (int(m[2]) == year + 1))]

                pl[f'match_{year}'] = pl[f'match_{year}'].mask(
                    (pl['dob_month'] == month) &
                    (~pl[cols].isin(values)).all(axis=1),
                    False)

            if not dask:
                nobs_dropped[year]['buyin'] = (
                    1 - (pl[f'match_{year}'].sum() / nobs))

        regex_string = r'^{}'.format(var) + r'\d{2}\d{4}$'
        regex = re.compile(regex_string).search
        pl = pl.drop([x for x in pl.columns if regex(x)], axis=1)

        return pl, nobs_dropped

    def _str_in_keep_vars(
            self, instr: str, keep_vars: List[Union[str, Pattern]]) -> bool:
        """Return True if string is in keep_vars
        """

        match = False
        for keep_var in keep_vars:
            if isinstance(keep_var, str):
                if instr == keep_var:
                    match = True
            elif isinstance(keep_var, re._pattern_type):
                if keep_var.search(instr):
                    match = True
        return match

    def get_cohort(
            self,
            gender: Optional[str] = None,
            ages: Union[int, List[int], None] = None,
            races: Union[str, List[str], None] = None,
            rti_race: bool = False,
            buyin_val: Union[str, List[str], None] = None,
            hmo_val: Union[str, List[str], None] = None,
            join: str = 'outer',
            keep_vars: Union[str, Pattern, List[Union[str, Pattern]],
                             None] = [],
            dask: bool = False,
            verbose: bool = False): # yapf: disable
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

        if self.verbose | verbose:
            verbose = True
            self.t0 = time()

        if self.dask or dask:
            dask = True

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
            msg = f"""\
            Starting cohort retrieval
            (`None` means no restriction)
            - percent sample: {self.percent}
            - years: {list(self.years)}
            - ages: {list(ages) if ages else None}
            - races: {races if races else None}
            - buyin values: {buyin_val}
            - HMO values: {hmo_val}
            - extra variables: {keep_vars}
            """
            print(_mywrap(msg))

        toload_vars = self._get_cohort_get_vars_toload(
            gender, ages, races, race_col, buyin_val, hmo_val, keep_vars)

        # Now perform extraction
        extracted_dfs = []
        nobs_dropped = {year: {} for year in self.years}

        # Do filtering for all vars that are checkable within a single year's
        # data
        for year in self.years:
            pl, nobs_dropped = self._get_cohort_extract_each_year(
                year=year,
                toload_vars=toload_vars[year],
                nobs_dropped=nobs_dropped,
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
            extracted_dfs.append(pl)

        if verbose & (len(extracted_dfs) > 1):
            msg = f"""\
            Merging together beneficiary files
            - years: {list(self.years)}
            - merge type: {join}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
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
            pl, nobs_dropped = self._get_cohort_age_year_filter(
                pl=pl,
                var='buyin',
                values=buyin_val,
                nobs_dropped=nobs_dropped,
                dask=dask,
                verbose=verbose)

        if (hmo_val is not None) and (self.year_type == 'age'):
            pl, nobs_dropped = self._get_cohort_age_year_filter(
                pl=pl,
                var='hmoind',
                values=hmo_val,
                nobs_dropped=nobs_dropped,
                dask=dask,
                verbose=verbose)

        if ((buyin_val is not None) or
            (hmo_val is not None)) and (self.year_type == 'age'):

            to_drop = ['dob_month']
            if not self._str_in_keep_vars('bene_dob', keep_vars):
                to_drop.append('bene_dob')

            pl = pl.drop(to_drop, axis=1)

        if self.year_type == 'age':
            # Do correct filtering of data based on desired join
            match_cols = [f'match_{year}' for year in self.years[:-1]]
            if join == 'inner':
                pl = pl.loc[pl[match_cols].all(axis=1)]
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
            - percent sample: {pct_dict[self.percent]}%
            - years: {list(self.years)}
            - ages: {list(ages) if ages else None}
            - races: {races if races else None}
            - buyin values: {buyin_val}
            - HMO values: {hmo_val}
            - extra variables: {keep_vars}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
            """
            print(_mywrap(msg))

    @staticmethod
    def _get_pattern(obj: Union[str, Pattern]) -> str:
        """
        If str, returns str. If compiled regex, returns string representation of
        that pattern
        """
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, re._pattern_type):
            return obj.pattern
        else:
            raise TypeError('Provided non string or regex to _get_pattern()')

    def _create_rename_dict_each(
            self,
            codes: List[Union[str, Pattern]],
            rename: Union[str, None, List[str], Dict[str, str]]
            ) -> Dict[str, str]: # yapf: disable
        """Create rename dictionary for single code at a time.

        Args:
            codes: codes to search for. In this function, this refers to only codes for _either_ `hcpcs`, `icd9_dx`, or `icd9_sg`.
            rename: how to rename codes.
        Returns:
            ``dict`` where keys are codes to match and values are new names for each.
        Raises:
            AssertionError if the length of ``rename`` list doesn't match the length of ``codes``
            TypeError if rename is not a str, dict, or list of str
        """
        # If rename is non missing, make sure codes is not None
        if rename is None:
            return {}
        if rename == '':
            return {}

        # Therefore length of codes should be >= 1
        assert codes is not None

        if isinstance(rename, str):
            # Assert length of codes is 1
            assert len(codes) == 1
            d = {self._get_pattern(codes): rename}
            return d

        if isinstance(rename, dict):
            # Make sure keys of dict are in codes
            all_codes = [self._get_pattern(code) for code in codes]
            msg = _mywrap(
                f"""\
            Keys of the inner rename dict need to be a subset of the codes provided to search through
            """)
            assert rename.keys() <= set(all_codes), msg
            return rename

        if isinstance(rename, list):
            # If the values of rename are lists, make sure they match up on length
            msg = f"""\
            If the values of the rename dictionary are lists, they need
            to match the length of the list of codes provided
            """
            msg = _mywrap(msg)
            assert len(codes) == len(rename), msg

            d = {self._get_pattern(code): y for code, y in zip(codes, rename)}
            return d

        msg = 'values of rename dict must be str, dict, or list of str'
        raise TypeError(msg)

    def _create_rename_dict(
            self,
            codes: Dict[str, List[Union[str, Pattern]]] = {},
            rename: Dict[str, Union[str, None, List[str], Dict[str, str]]] = {}
            ) -> Dict[str, str]: # yapf: disable
        """Make dictionary where the keys are codes/pattern strings and values
        are new column names

        Args:
            codes: dict of codes provided to :func:`search_for_codes`
                {'hcpcs': hcpcs_codes,
                'icd9_dx': icd9_dx_codes,
                'icd9_sg': icd9_sg_codes}
            rename: dict to describe how to rename extracted variables
                - keys must be `'hcpcs'`, `'icd9_dx'`, `'icd9_sg'`
                - values for each can either be str, dict, or list of str.

        Returns:
            ``dict`` where keys are codes to match and values are new names for each.
        Raises:
            ValueError if the keys of ``rename`` dict are not a subset of 'hcpcs', 'icd9_dx', and 'icd9_sg'.

        """

        rename_new = []
        for var in ['hcpcs', 'icd9_dx', 'icd9_sg']:
            if rename.get(var) is not None:
                d = self._create_rename_dict_each(codes[var], rename[var])
                rename_new.append(d)
            else:
                rename_new.append({})

        # Assert all keys are unique
        keys = [list(x.keys()) for x in rename_new]
        keys = [item for sublist in keys for item in sublist]
        msg = 'Codes given must be unique'
        assert len(keys) == len(set(keys)), msg

        # Assert all values are unique
        vals = [list(x.values()) for x in rename_new]
        vals = [item for sublist in vals for item in sublist]
        msg = 'Values of rename dict must be unique'
        assert len(vals) == len(set(vals)), msg

        return {k: v for d in rename_new for k, v in d.items()}

    class _ReturnSearchForCodesTypeCheck(NamedTuple):
        data_types: List[str]
        pl_ids_to_filter: Optional[pd.DataFrame]
        codes: Dict[str, List[Union[str, Pattern]]]
        icd9_dx_max_cols: Optional[int]
        icd9_sg_max_cols: Optional[int]
        keep_vars: Dict[str, List[Union[str, Pattern]]]
        collapse_codes: bool
        rename: Dict[str, Union[str, List[str], Dict[str, str], None]]
        convert_ehic: bool
        dask: bool
        verbose: bool

    def _search_for_codes_type_check(
            self,
            data_types: Union[str, List[str]],
            pl: Optional[pd.DataFrame],
            hcpcs: Union[str, Pattern, List[Union[str, Pattern]], None],
            icd9_dx: Union[str, Pattern, List[Union[str, Pattern]], None],
            icd9_dx_max_cols: Optional[int],
            icd9_sg: Union[str, Pattern, List[Union[str, Pattern]], None],
            icd9_sg_max_cols: Optional[int],
            keep_vars: Dict[str, Union[str, Pattern, List[Union[str, Pattern]], None]],
            collapse_codes: bool,
            rename: Dict[str, Union[str, List[str], Dict[str, str], None]],
            convert_ehic: bool,
            dask: bool,
            verbose: bool) -> _ReturnSearchForCodesTypeCheck: # yapf: disable
        """Check types and valid values for :func:`search_for_codes`

        Also resolves input into correct value
        """

        ok_data_types = ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']

        if data_types is None:
            raise TypeError('data_types cannot be None')
        if isinstance(data_types, str):
            data_types = [data_types]
        if (data_types == []) | (data_types == ['']):
            raise TypeError('data_types cannot be empty')
        if isinstance(data_types, list):
            # Check that all data types provided to search through exist
            if not set(data_types).issubset(ok_data_types):
                invalid_vals = list(set(data_types).difference(ok_data_types))
                msg = f"""\
                {invalid_vals} does not match any dataset.
                Allowed `data_types` are {ok_data_types}.
                """
                raise ValueError(_mywrap(msg))
        else:
            raise TypeError('data_types must be str or List[str]')

        if pl is not None:
            if not isinstance(pl, pd.DataFrame):
                raise TypeError('pl must be DataFrame')

            columns = [*pl.columns, pl.index.name]
            if not any(x in columns for x in ['ehic', 'bene_id']):
                raise ValueError('pl must have `ehic` or `bene_id` as a column')

            pl_ids_to_filter = pl.reset_index()[list({
                'ehic', 'bene_id'}.intersection(columns))].copy()

        # Assert that keep_vars is a dict and that the keys are in ok_data_types
        if not isinstance(keep_vars, dict):
            raise TypeError('keep_vars must be dict')
        # Initialize key for each data_type given if they don't already exist
        for data_type in data_types:
            keep_vars[data_type] = keep_vars.get(data_type, [])
        if not set(keep_vars.keys()).issubset(ok_data_types):
            invalid_vals = list(set(keep_vars.keys()).difference(ok_data_types))
            msg = f"""\
            {invalid_vals} does not match any dataset.
            Allowed keys of `keep_vars` are {ok_data_types}.
            """
            raise ValueError(_mywrap(msg))

        # Coerce values of keep_vars to List[Union[str, Pattern]]
        msg = f"""\
        keep_vars must be str, compiled regex, or List[str, compiled regex]
        """
        for k, v in keep_vars.items():
            if v is None:
                keep_vars[k] = []
            if isinstance(v, (str, re._pattern_type)):
                keep_vars[k] = [v]
            elif isinstance(v, list):
                if not all(isinstance(x, (str, re._pattern_type)) for x in v):
                    raise TypeError(_mywrap(msg))
            else:
                raise TypeError(_mywrap(msg))

        codes = {'hcpcs': hcpcs, 'icd9_dx': icd9_dx, 'icd9_sg': icd9_sg}

        msg = f"""\
        Codes to search through must be str, compiled regex, or
        List[str, compiled regex]
        """
        all_codes = []
        for name, code in codes.items():
            if code is None:
                codes[name] = []
                continue
            if isinstance(code, (str, re._pattern_type)):
                code = [code]
            elif isinstance(code, list):
                # Check all elements of list are either str or Pattern
                if not all(isinstance(x, (str, re._pattern_type))
                           for x in code):
                    raise TypeError(_mywrap(msg))
            else:
                raise TypeError(_mywrap(msg))

            codes[name] = code
            all_codes.extend(code)

        if not isinstance(collapse_codes, bool):
            raise TypeError('collapse_codes must be bool')
        if not isinstance(convert_ehic, bool):
            raise TypeError('convert_ehic must be bool')
        if not isinstance(dask, bool):
            raise TypeError('dask must be type bool')
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be bool')
        if not isinstance(rename, dict):
            raise TypeError('rename must be dict')

        # Only allowed keys of rename are 'hcpcs', 'icd9_dx', and 'icd9_sg'
        if not rename.keys() <= set(['hcpcs', 'icd9_dx', 'icd9_sg']):
            msg = f"""\
            Only allowed keys of rename dict are 'hcpcs', 'icd9_dx', and
            'icd9_sg'
            """
            raise ValueError(_mywrap(msg))

        if not collapse_codes:
            all_codes = [self._get_pattern(x) for x in all_codes]
            msg = 'Code patterns given must be unique'
            if not len(all_codes) == len(set(all_codes)):
                raise ValueError(msg)

        if collapse_codes and any([x is not None for x in rename.values()]):
            msg = f"""\
            rename argument not allowed when collapse_codes is True
            """
            raise ValueError(_mywrap(msg))

        if (codes['icd9_dx'] == []) and (icd9_dx_max_cols is not None):
            msg = f"""\
            icd9_dx_max_cols argument not allowed when icd9_dx is None
            """
            raise ValueError(_mywrap(msg))
        if (codes['icd9_sg'] == []) and (icd9_sg_max_cols is not None):
            msg = f"""\
            icd9_sg_max_cols argument not allowed when icd9_sg is None
            """
            raise ValueError(_mywrap(msg))

        return self._ReturnSearchForCodesTypeCheck(
            data_types=data_types,
            pl_ids_to_filter=pl_ids_to_filter,
            codes=codes,
            icd9_dx_max_cols=icd9_dx_max_cols,
            icd9_sg_max_cols=icd9_sg_max_cols,
            keep_vars=keep_vars,
            collapse_codes=collapse_codes,
            rename=rename,
            convert_ehic=convert_ehic,
            dask=dask,
            verbose=verbose)

    def search_for_codes(
            self,
            data_types: Union[str, List[str]],
            pl: Optional[pd.DataFrame] = None,
            hcpcs: Union[str, Pattern, List[Union[str, Pattern]], None] = None,
            icd9_dx: Union[str, Pattern, List[Union[str, Pattern]], None] = None,
            icd9_dx_max_cols: Optional[int] = None,
            icd9_sg: Union[str, Pattern, List[Union[str, Pattern]], None] = None,
            icd9_sg_max_cols: Optional[int] = None,
            keep_vars: Dict[str, Union[str, Pattern, List[Union[str, Pattern]], None]] = {},
            collapse_codes: bool = True,
            rename: Dict[str, Union[str, List[str], Dict[str, str], None]] = {
                'hcpcs': None,
                'icd9_dx': None,
                'icd9_sg': None},
            convert_ehic: bool = True,
            dask: bool = False,
            verbose: bool = False): # yapf: disable
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
            pl:
                Patient-level DataFrame used to filter cohort before searching
                code columns. Unnecessary if :func:`get_cohort` is called before
                this. Must have at least ``ehic`` or ``bene_id`` as a column.
                search through. If ``None``, will search through all columns.
            icd9_sg: ICD-9 procedure codes to search for
            icd9_sg_max_cols: Max number of ICD9 procedure code columns to
                search through. If ``None``, will search through all columns.
            keep_vars: column names to return
            collapse_codes: If ``True``, returns a single column named
                ``match`` that is ``True`` for the claims with a matched code and ``False`` otherwise. If ``collapse_codes`` is ``False``, returns a column for each code provided
            rename: Match columns to rename when ``collapse_codes`` is ``False``.
            convert_ehic: If ``True``, merges on ``bene_id`` for years <
                2006
            dask: Use dask library for out of core computation. Not yet implemented; as of now everything happens in core.
            verbose: Print progress of program to console

        Returns:
            Creates ``.cl`` attribute. This is a dict where keys are the
            data_types provided and values are pandas DataFrames with
            ``bene_id`` as index and indicator columns for each code provided.
        """

        if self.verbose or verbose:
            verbose = True
            self.t0 = time()

        objs = self._search_for_codes_type_check(
            data_types=data_types,
            pl=pl,
            hcpcs=hcpcs,
            icd9_dx=icd9_dx,
            icd9_dx_max_cols=icd9_dx_max_cols,
            icd9_sg=icd9_sg,
            icd9_sg_max_cols=icd9_sg_max_cols,
            keep_vars=keep_vars,
            collapse_codes=collapse_codes,
            rename=rename,
            convert_ehic=convert_ehic,
            dask=dask,
            verbose=verbose)
        data_types = objs.data_types
        pl_ids_to_filter = objs.pl_ids_to_filter
        codes = objs.codes
        icd9_dx_max_cols = objs.icd9_dx_max_cols
        icd9_sg_max_cols = objs.icd9_sg_max_cols
        keep_vars = objs.keep_vars
        collapse_codes = objs.collapse_codes
        rename = objs.rename
        convert_ehic = objs.convert_ehic
        dask = objs.dask
        verbose = objs.verbose

        if self.dask or dask:
            dask = True

        # Dask isn't ready yet
        dask = False

        ok_data_types = {
            'hcpcs': {'carl', 'ipr', 'opr'},
            'icd9_dx': {'carc', 'carl', 'ipc', 'med', 'opc'},
            'icd9_sg': {'ipc', 'med', 'opc'}}
        codes_fmt = {
            'hcpcs': 'HCPCS',
            'icd9_dx': 'ICD-9 diagnosis',
            'icd9_sg': 'ICD-9 procedure'}

        # Print which codes are searched in which dataset
        if verbose and any(v is not None for v in codes.values()):
            msg = f"""\
            Will check the following codes
            - percent sample: {pct_dict[self.percent]}%
            - years: {list(self.years)}
            """
            msg = _mywrap(msg)
            for k, v in codes.items():
                if v != []:
                    dts = list(set(data_types) & ok_data_types[k])
                    if len(set(dts)) > 0:
                        msg += _mywrap(f"""\
                        - {codes_fmt[k]} codes: {v}
                          in data types: {dts}
                        """) # yapf: disable

            print(msg)

        if not all([x is None for x in rename.values()]):
            rename = self._create_rename_dict(codes=codes, rename=rename)
        else:
            rename = {}

        data = {}
        for data_type in data_types:
            data[data_type] = {}
            for year in self.years:
                if verbose:
                    msg = _mywrap(f"""\
                    Searching for codes
                    - percent sample: {pct_dict[self.percent]}%
                    - year: {year}
                    - data type: {data_type}
                    """) # yapf: disable
                    for k, v in codes.items():
                        if data_type in ok_data_types[k]:
                            if v != []:
                                msg += _mywrap(f"""\
                                - {codes_fmt[k]} codes: {v}
                                """) # yapf: disable

                    if keep_vars[data_type] != []:
                        msg += _mywrap(f"""\
                        - Keeping variables: {keep_vars[data_type]}
                        """) # yapf: disable
                    msg += _mywrap(f"""\
                    - time in function: {(time() - self.t0) / 60:.2f} minutes
                    - time in class: {(time() - self.tc) / 60:.2f} minutes
                    """) # yapf: disable
                    print(msg)

                data[data_type][year] = self._search_for_codes_single_year(
                    year=year,
                    data_type=data_type,
                    pl_ids_to_filter=pl_ids_to_filter,
                    codes=codes,
                    icd9_dx_max_cols=icd9_dx_max_cols,
                    icd9_sg_max_cols=icd9_sg_max_cols,
                    keep_vars=keep_vars[data_type],
                    rename=rename,
                    collapse_codes=collapse_codes,
                    dask=dask,
                    verbose=verbose)

        self.cl = self._search_for_codes_data_join(
            data=data, convert_ehic=convert_ehic, verbose=verbose)

        if verbose:
            msg = f"""\
            Finished searching for codes
            - percent sample: {self.percent}
            - years: {list(self.years)}
            - data_types: {data_types}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        return

    def _search_for_codes_data_join(self,
            data: Dict[str, Dict[int, pd.DataFrame]],
            convert_ehic: bool = True,
            verbose: bool = False) -> pd.DataFrame: # yapf: disable
        """Join year-data_type codes datasets.

        Args:
            data: dict of dataframes from _search_for_codes_single_year
            convert_ehic: If true, convert ehic to bene_id
            verbose: Print logging messages
        """

        if verbose:
            msg = f"""\
            Concatenating matched codes across years
            - years: {list(self.years)}
            - data types: {list(data.keys())}
            - time in function: {(time() - self.t0) / 60:.2f} minutes
            - time in class: {(time() - self.tc) / 60:.2f} minutes
            """
            print(_mywrap(msg))

        years_ehic = [x for x in self.years if x < 2006]
        years_bene_id = [x for x in self.years if x >= 2006]

        if years_bene_id:
            for data_type in data.keys():
                data[data_type]['bene_id'] = pd.concat(
                    [data[data_type].pop(year) for year in years_bene_id],
                    sort=False)

        # Don't go through ehic process if data is only post 2006
        if not years_ehic:
            for data_type in data.keys():
                data[data_type] = data[data_type].pop('bene_id')
            return data

        # Always convert ehic to bene_id if data from before *and* after 2006
        if years_ehic and years_bene_id:
            convert_ehic = True

        # Concatenate ehic data (2005 and earlier)
        if not convert_ehic:
            for data_type in data.keys():
                data[data_type]['ehic'] = pd.concat(
                    [data[data_type].pop(year) for year in years_ehic],
                    sort=False)
        else:
            for year in years_ehic:
                # Get ehic-bene_id crosswalk
                # If self.pl exists, then cl data frames use only those ids
                # So I can merge using that
                if self.pl is not None:
                    if f'match_{year}' in self.pl.columns:
                        right = self.pl.loc[self.pl[f'match_{year}'], 'ehic']
                    else:
                        right = self.pl['ehic']
                    right = right.to_frame()
                else:
                    if self.parquet_engine == 'pyarrow':
                        pf = pq.ParquetFile(
                            self._fpath(self.percent, year, 'bsfab'))
                        right = pf.read(
                            columns=['ehic'],
                            nthreads=2).to_pandas().set_index('bene_id')
                    elif self.parquet_engine == 'fastparquet':
                        pf = fp.ParquetFile(
                            self._fpath(self.percent, year, 'bsfab'))
                        right = pf.to_pandas(columns=['ehic'], index='bene_id')

                # Join bene_ids onto data using ehic
                for data_type in data.keys():
                    data[data_type][year] = data[data_type][year].merge(
                        right, how='left', left_index=True, right_on='ehic')

            # Concatenate ehic data
            for data_type in data.keys():
                data[data_type]['ehic'] = pd.concat(
                    [data[data_type].pop(year) for year in years_ehic],
                    sort=False)

        for data_type in data.keys():
            data[data_type] = pd.concat(
                [data[data_type].pop('ehic'), data[data_type].pop('bene_id')],
                sort=False)

        return data

    def _search_for_codes_df_inner(self,
            cl: Union[pd.DataFrame, dd.DataFrame],
            codes: Dict[str, List[Union[str, Pattern]]],
            cols: Dict[str, Union[str, List[str]]],
            keep_vars: List[Union[str, Pattern]],
            rename: Dict[str, str],
            collapse_codes: bool,
            pl_ids_to_filter: Optional[pd.Index],
            ) -> Union[pd.DataFrame, dd.DataFrame]: # yapf: disable
        """Meat of the code to search for codes in files

        Dask.dataframe doesn't support multiindexes yet (see
        github.com/dask/dask/issues/811, github.com/dask/dask/issues/1493). So
        as of now, the loaded cl data is singly-indexed with the patient-level
        identifier.

        Args:
            cl: claim-level data. Index is ``cols['pl_id']``, i.e. either
                ``ehic`` for pre-2006 or ``bene_id`` for post-2006.
            codes: Keys are 'hcpcs', 'icd9_dx', and 'icd9_sg'. Values are lists
                with codes to search for.
            cols:
                - ``cl_id``: unique-identifying claim-level variable. i.e.
                    ``medparid``
                - ``pl_id``: unique-identifying patient-level variable. Either
                    ``bene_id`` or ``ehic``.
                - ``hcpcs``, ``icd9_dx``, or ``icd9_sg``: columns to search over
            keep_vars: List of variables to keep in returned dataset
            rename: Dict to rename variables; key is old name, value is new name
            collapse_codes: Whether to return match column for each code
            pl_ids_to_filter: Index of either ``bene_id``s or ``ehic``s, derived from the result of :func:``get_cohort``.

        Returns:
            Data with boolean match columns instead of code columns.
        """
        if pl_ids_to_filter is not None:
            index_name = cl.index.name
            cl = cl.join(pd.DataFrame(index=pl_ids_to_filter), how='inner')
            cl.index.name = index_name

        if not any(v is not None for v in codes.values()):
            return cl

        # The index needs to be unique for the stuff I do below with first
        # saving all indices in a var idx, then using that with cl.loc[].
        # If index is bene_id, it'll set matched to true for _anyone_ who
        # had a match _sometime_.
        cl = cl.reset_index().set_index(cols['cl_id'])

        if collapse_codes:
            cl['match'] = False
        else:
            all_created_cols = []

        for key, val in codes.items():
            # If no columns to search over; move to next iteration of loop
            if cols[key] == []:
                continue

            for code in val:
                if collapse_codes:
                    if isinstance(code, re._pattern_type):
                        cl.loc[cl[cols[key]].apply(
                            lambda col: col.str.contains(code)).any(
                                axis=1), 'match'] = True
                        cl[cols[key]].apply(
                            lambda col: col.str.contains(code)).any(axis=1)
                    else:
                        cl.loc[(
                            cl[cols[key]] == code).any(axis=1), 'match'] = True
                else:
                    cl[self._get_pattern(code)] = False
                    if isinstance(code, re._pattern_type):
                        idx = cl.index[cl[cols[key]].apply(
                            lambda col: col.str.contains(code)).any(axis=1)]
                    else:
                        idx = cl.index[(cl[cols[key]] == code).any(axis=1)]
                    cl.loc[idx, self._get_pattern(code)] = True
                    all_created_cols.append(self._get_pattern(code))

            # cols[key] only includes the variables for the specific codes I'm
            # looking at, so should be fine within the loop.
            cols_todrop = [x for x in cols[key] if x not in cols['keep_vars']]
            cl = cl.drop(cols_todrop, axis=1)

        if not collapse_codes:
            cl['match'] = (cl[all_created_cols] == True).any(axis=1)

            # Rename columns according to `rename` dictionary
            cl = cl.rename(columns=rename)

        # Keep all rows; not just matches
        # TODO probably want to add a switch here to allow for people
        # to extract just matches if desired.
        cl = cl.reset_index().set_index(cols['pl_id'])

        return cl

    def _search_for_codes_single_year(
            self,
            year: int,
            data_type: str,
            pl_ids_to_filter: Optional[pd.DataFrame],
            codes: Dict[str, List[Union[str, Pattern]]],
            icd9_dx_max_cols: Optional[int],
            icd9_sg_max_cols: Optional[int],
            keep_vars: List[Union[str, Pattern]],
            rename: Dict[str, str],
            collapse_codes: bool,
            dask: bool,
            verbose:bool) -> pd.DataFrame: # yapf: disable
        """Search in a single claim-level dataset for HCPCS/ICD9 codes

        Note: Each code given must be distinct, or collapse_codes must be True

        Args:
            year: year of data to search
            data_type: One of carc, carl, ipc, ipr, med, opc, opr
            pl_ids_to_filter: user-provided dataframe with ``bene_id`` and/or
                ``ehic``. Allows for bypassing of get_cohort().
            codes: dict of codes to look for
            icd9_dx_max_cols: Max number of ICD9 diagnosis code columns to
                search through
            icd9_sg_max_cols: Max number of ICD9 procedure code columns to
                search through
            keep_vars: list of column names to return
            rename: dictionary where keys are codes to match, and values are
                new column names
            collapse_codes: If True, returns a single column "match";
                else it returns a column for each code provided
            dask: Use dask library for out of core computation
            verbose: Print logging messages to console

        Returns:
            DataFrame with bene_id and bool columns for each code to search for
        """

        # Determine which variables to extract
        if self.parquet_engine == 'pyarrow':
            all_cols = pq.ParquetFile(
                self._fpath(self.percent, year, data_type)).schema.names
        elif self.parquet_engine == 'fastparquet':
            all_cols = fp.ParquetFile(
                self._fpath(self.percent, year, data_type)).columns

        icd9_sg_regex = r'^icd_prcdr_cd(\d+)$'
        if data_type == 'carl':
            icd9_dx_regex = r'icd_dgns_cd(\d*)$'
        elif data_type == 'med':
            icd9_dx_regex = r'^dgnscd(\d+)$$'
        else:
            icd9_dx_regex = r'^icd_dgns_cd(\d+)$'
        cols: Dict[str, List[str]] = {
            'cl_id': [
                x for x in all_cols
                if re.search(r'^medparid$|^clm_id$|^claimindex$', x)],
            'pl_id': ['ehic'] if year < 2006 else ['bene_id'],
            'keep_vars': [
                x for x in all_cols if self._str_in_keep_vars(x, keep_vars)]}
        if codes['hcpcs']:
            cols['hcpcs'] = [x for x in all_cols if re.search(r'^hcpcs_cd$', x)]
        else:
            cols['hcpcs'] = []

        if codes['icd9_sg']:
            cols['icd9_sg'] = [
                x for x in all_cols if re.search(icd9_sg_regex, x)]
        else:
            cols['icd9_sg'] = []

        if codes['icd9_dx']:
            cols['icd9_dx'] = [
                x for x in all_cols if re.search(icd9_dx_regex, x)]
        else:
            cols['icd9_dx'] = []

        # Check cols against keep_vars
        # Is there an item in keep_vars that wasn't matched?
        # NOTE need to check this against regex values of keep_vars
        for var in cols['keep_vars']:
            if [x for x in all_cols if re.search(var, x)] == []:
                msg = f"""\
                WARNING: variable `{var}` in the keep_vars argument
                was not found in {data_type}
                """
                print(_mywrap(msg))

        if icd9_dx_max_cols is not None:
            cols['icd9_dx'] = [
                x for x in all_cols for m in [re.search(icd9_dx_regex, x)] if m
                if int(m[1]) <= icd9_dx_max_cols]
        if icd9_sg_max_cols is not None:
            cols['icd9_sg'] = [
                x for x in all_cols for m in [re.search(icd9_sg_regex, x)] if m
                if int(m[1]) <= icd9_sg_max_cols]

        cols_toload = set(item for subl in cols.values() for item in subl)
        # Now that list flattening is over, make 'cl_id' and 'pl_id' strings
        # instead of list of string
        for i in ['cl_id', 'pl_id']:
            assert len(cols[i]) == 1
            cols[i] = cols[i][0]

        if pl_ids_to_filter is None:
            # Assumes bene_id or ehic is index name or name of a column
            # Unless `join` in `get_cohort` is `inner`, we have a variable
            # `match_{year}` that's True is the patient was found in that year
            # and False otherwise. We should use that information so that we
            # aren't trying to join observations that we know don't exist.
            if self.pl is not None:
                if (f'match_{year}' in self.pl.columns):
                    if cols['pl_id'] == self.pl.index.name:
                        pl_ids_to_filter = self.pl.index[self.pl[f'match_{year}']]
                    else:
                        pl_ids_to_filter = pd.Index(self.pl.loc[self.pl[f'match_{year}'], cols['pl_id']].values)
                else:
                    if cols['pl_id'] == self.pl.index.name:
                        pl_ids_to_filter = self.pl.index
                    else:
                        pl_ids_to_filter = pd.Index(self.pl[cols['pl_id']].values)

        path = self._fpath(self.percent, year, data_type)
        if dask:
            # NOTE: should the index here be cols['cl_id'] ?
            cl = dd.read_parquet(
                path,
                columns=cols_toload - set([cols['pl_id']]),
                index=cols['pl_id'])
        elif self.parquet_engine == 'pyarrow':
            pf = pq.ParquetFile(path)
            itr = (
                pf.read_row_group(
                    i,
                    columns=cols_toload,
                    nthreads=min(len(cols_toload),
                                 self.parquet_nthreads)).to_pandas().set_index(
                                     cols['pl_id'])
                for i in range(pf.num_row_groups))
        elif self.parquet_engine == 'fastparquet':
            pf = fp.ParquetFile(path)
            itr = pf.iter_row_groups(columns=cols_toload, index=cols['pl_id'])

        if dask:
            cl = self._search_for_codes_df_inner(
                cl=cl,
                codes=codes,
                cols=cols,
                keep_vars=keep_vars,
                rename=rename,
                collapse_codes=collapse_codes,
                pl_ids_to_filter=pl_ids_to_filter)
        else:
            # This holds the df's from each iteration over the claim-level
            # dataset
            all_cl: List[pd.DataFrame] = []
            for cl in itr:
                cl = self._search_for_codes_df_inner(
                    cl=cl,
                    codes=codes,
                    cols=cols,
                    keep_vars=keep_vars,
                    rename=rename,
                    collapse_codes=collapse_codes,
                    pl_ids_to_filter=pl_ids_to_filter)
                all_cl.append(cl)

            cl = pd.concat(all_cl, axis=0, sort=False)

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
        #     left_on=cols['pl_id'],
        #     right_index=True).set_index(cols['pl_id'])

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

    def to_stata(self, attr: str, **kwargs):
        """Wrapper to export to stata.

        Will automatically add

        Args:
            attr : str
                either 'pl' or 'cl.med', 'cl.opc', 'cl.opr', etc.
            fname : path (string), buffer or path object
                string, path object (pathlib.Path or py._path.local.LocalPath) or
                object implementing a binary write() functions. If using a buffer
                then the buffer will not be automatically closed after the file
                data has been written.
            convert_dates : dict
                Dictionary mapping columns containing datetime types to stata
                internal format to use when writing the dates. Options are 'tc',
                'td', 'tm', 'tw', 'th', 'tq', 'ty'. Column can be either an integer
                or a name. Datetime columns that do not have a conversion type
                specified will be converted to 'tc'. Raises NotImplementedError if
                a datetime column has timezone information.
            encoding : str
                Default is latin-1. Unicode is not supported.
            time_stamp : datetime
                A datetime to use as file creation date.  Default is the current
                time.
            data_label : str
                A label for the data set.  Must be 80 characters or smaller.
            variable_labels : dict
                Dictionary containing columns as keys and variable labels as
                values. Each label must be 80 characters or smaller.
            version : {114, 117}
                Version to use in the output dta file.  Version 114 can be used
                read by Stata 10 and later.  Version 117 can be read by Stata 13
                or later. Version 114 limits string variables to 244 characters or
                fewer while 117 allows strings with lengths up to 2,000,000
                characters.
            convert_strl : list, optional
                List of column names to convert to string columns to Stata StrL
                format. Only available if version is 117.  Storing strings in the
                StrL format can produce smaller dta files if strings have more than
                8 characters and values are repeated.

        Examples:
            >>> mdf.to_stata(attr='pl', fname='patient_level_file.dta')
            >>> mdf.to_stata(attr='cl.med', fname='medpar_extract.dta')
            Or with dates
            >>> mdf.to_stata(attr='cl.med', fname='medpar_extract.dta',
                             convert_dates={'admsndt': 'td'})
        """

        data_label_dict = {
            'pl': 'Patient-level',
            'cl.med': 'Claim-level MedPAR',
            'cl.carc': 'Claim-level Carrier Claims',
            'cl.carl': 'Claim-level Carrier Line',
            'cl.ipc': 'Claim-level Inpatient Claims',
            'cl.ipr': 'Claim-level Inpatient Revenue Center',
            'cl.opc': 'Claim-level Outpatient Claims',
            'cl.opr': 'Claim-level Outpatient Revenue Center'}

        data_type = re.search(r'\.(.+)', attr)[1]
        if attr == 'pl':
            data = self.pl
        else:
            data = self.cl[data_type]

        columns = [*list(data.columns), data.index.name]
        var_labels = {
            col: codebook(data_type)[col]['name']
            for col in columns
            if col in codebook(data_type).keys()}
        kwargs['variable_labels'] = {
            **var_labels,
            **kwargs.get('variable_labels', {})}

        kwargs['write_index'] = True
        kwargs['data_label'] = kwargs.get(
            'data_label', f'{data_label_dict[attr]} data extract.')

        data.to_stata(**kwargs)
        return
