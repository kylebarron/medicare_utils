#! /usr/bin/env python3
"""Main module."""

import re
import pandas as pd
import fastparquet as fp
import numpy as np
import pyarrow.parquet as pq
from time import time
from multiprocessing import cpu_count

from .utils import fpath, mywrap

allowed_pcts = ['0001', '01', '05', '20', '100']
pct_dict = {0.01: '0001', 1: '01', 5: '05', 20: '20', 100: '100'}


class MedicareDF(object):
    """A class to organize Medicare operations"""

    def __init__(
            self,
            percent,
            years,
            verbose=False,
            parquet_engine='pyarrow',
            parquet_nthreads=None,
            dta_path: str = '/disk/aging/medicare/data',
            pq_path: str = '/homes/nber/barronk/agebulk1/raw/pq'):
        """Return a MedicareDF object

        Attributes:
            percent (str, int, or float): percent sample of data to use
            years (list[int]): years of data to use
            verbose (bool): Print status of program
            parquet_engine (str): 'pyarrow' or 'fastparquet'
            parquet_nthreads (int): number of threads to use when reading file
        """

        if (type(percent) == float) or (type(percent) == int):
            try:
                self.percent = pct_dict[percent]
            except KeyError:
                msg = f"""\
                percent provided is not valid.
                Valid arguments are: {list(pct_dict.keys())}
                """
                raise ValueError(mywrap(msg))
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

        assert min(years) >= 2001
        assert max(years) <= 2015

        self.years = years
        self.verbose = verbose

        if parquet_engine not in ['pyarrow', 'fastparquet']:
            raise ValueError('parquet_engine must be pyarrow or fastparquet')

        self.parquet_engine = parquet_engine

        if parquet_nthreads is None:
            parquet_nthreads = cpu_count()

        self.parquet_nthreads = parquet_nthreads

        self.pl = None
        self.cl = None

        self.dta_path = dta_path
        self.pq_path = pq_path

    def fpath(self, percent: str, year: int, data_type: str, dta: bool = False):

        return fpath(
            percent=percent,
            year=year,
            data_type=data_type,
            dta=dta,
            dta_path=self.dta_path,
            pq_path=self.pq_path)

    def _get_variables_to_import(self, year, data_type, import_vars):
        """Get list of variable names to import from given file

        NOTE Not currently used

        Returns:
            List of strings of variable names to import from file
        """

        if type(year) != int:
            raise TypeError('year must be type int')

        allowed_data_types = [
            'carc', 'carl', 'den', 'ipc', 'ipr', 'med', 'opc', 'opr', 'bsfab',
            'bsfcc', 'bsfcu', 'bsfd']
        if data_type not in allowed_data_types:
            msg = f'data_type must be one of: {allowed_data_types}'
            raise ValueError(msg)

        import_vars = list(set(import_vars))

        cols = fp.ParquetFile(self.fpath(self.percent, year, data_type)).columns
        tokeep_list = []

        for var in import_vars[:]:
            # Keep columns that match text exactly
            if var in cols:
                tokeep_list.append(var)
                import_vars.remove(var)

            # Then perform regex against other variables
            # else:
            #     re.search

    def get_cohort(
            self,
            gender=None,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=None,
            buyin_months=None,
            hmo_val=None,
            hmo_months=None,
            join='default',
            keep_vars=[],
            verbose=False):
        """Get cohort in standardized way

        Merges in such a way that age has to be within `ages` in any such year.
        Creates '.pl' attribute with patient-level data in the form of a
        pandas DataFrame. Index of returned DataFrame is always 'bene_id'.
        In pre-2006 years, 'ehic' will always be returned as a column.

        Args:
            gender (str): 'M', 'F', 'Male', 'Female', or None (keep both)
            ages (range, list[int], int):
                Minimum and maximum possible ages (inclusive)
            races (list[str], str): which races to include
            rti_race (bool): Whether to use the Research Triangle
                Institute race code
            buyin_val (list[str], str): The values `buyin\d\d` can take
            buyin_months (str): 'All', 'age_year'
                If 'age_year', years cannot be int
            join (str): method for joining across years
                Default is "outer" join for all years up to N-1, "left" for N
                Otherwise must be "left", "inner", "outer"
            keep_vars (list[str]): Variable names to keep in final output
            verbose (bool): Print status of program

        Returns:
            Creates attributes:
            - 'pl' with patient-level data in pandas DataFrame.
            - 'nobs_dropped' with dict of percent of observations dropped
                due to each filter.
        """

        if self.verbose:
            verbose = True

        if verbose:
            t0 = time()
            msg = f"""\
            Starting cohort retrieval
            - percent sample: {self.percent}
            - years: {self.years}
            - ages: {list(ages) if ages else None}
            - races: {list(races) if races else None}
            - buyin values: {buyin_val}
            - buyin months: {buyin_months}
            - HMO values: {hmo_val}
            - HMO months: {hmo_months}
            - extra variables: {keep_vars}
            """
            print(mywrap(msg))

        if len(self.years) == 1:
            if buyin_months == 'age_year':
                msg = "buyin_months can't be 'age_year' when one year is given"
                raise ValueError(msg)
        if type(ages) == int:
            ages = [ages]

        race_codebook = {
            'Unknown': '0',
            'White': '1',
            'Black': '2',
            'Other': '3',
            'Asian': '4',
            'Hispanic': '5',
            'North American Native': '6',
            'UNKNOWN': '0',
            'BLACK (OR AFRICAN-AMERICAN)': '2',
            'OTHER': '3',
            'ASIAN/PACIFIC ISLANDER': '4',
            'HISPANIC': '5',
            'AMERICAN INDIAN / ALASKA NATIVE': '6'}

        race_col = 'rti_race_cd' if rti_race else 'race'

        if type(races) == int:
            races = [str(races)]
        elif type(races) == str:
            regex = re.compile(races, re.IGNORECASE).search
            races = [val for key, val in race_codebook.items() if regex(key)]
            races = list(set(races))
            assert len(races) == 1
        elif type(races) == list:
            assert all((type(x) == str) or (type(x) == int) for x in races)

            races_new = []
            for race in races:
                if type(race) == str:
                    regex = re.compile(race, re.IGNORECASE).search
                    race = [
                        val for key, val in race_codebook.items() if regex(key)]
                    race = list(set(race))
                    assert len(race) == 1
                    races_new.append(race[0])
                else:
                    races_new.append(str(race))

            races = races_new

        buyin_val = [buyin_val] if type(buyin_val) == str else buyin_val
        allowed_buyin_months = ['all', 'age_year', None]
        if buyin_months not in allowed_buyin_months:
            msg = f'buyin_months must be one of: {allowed_buyin_months[:2]}'
            raise ValueError(msg)

        hmo_val = [hmo_val] if type(hmo_val) == str else hmo_val
        allowed_hmo_months = ['all', 'age_year', None]
        if hmo_months not in allowed_hmo_months:
            msg = f'hmo_months must be one of: {allowed_hmo_months[:2]}'
            raise ValueError(msg)

        allowed_join = ['default', 'left', 'inner', 'outer']
        if join not in allowed_join:
            msg = f'join must be one of: {allowed_join}'
            raise ValueError(msg)

        keep_vars = [keep_vars] if type(keep_vars) == str else keep_vars

        # Get list of variables to import for each year
        if ('age' in keep_vars) & (len(self.years) > 1):
            msg = """\
            Warning: Can't export age variable, exporting bene_dob instead
            """
            print(mywrap(msg))

            keep_vars.remove('age')
            keep_vars.append('bene_dob')

        tokeep_regex = []
        tokeep_regex.extend([r'^(ehic)$', r'^(bene_id)$'])
        if gender is not None:
            tokeep_regex.append(r'^(sex)$')
        if ages is not None:
            tokeep_regex.append(r'^(age)$')
        if races is not None:
            tokeep_regex.append(r'^({})$'.format(race_col))
        if buyin_val is not None:
            tokeep_regex.append(r'^(buyin\d{2})$')
            if buyin_months == 'age_year':
                tokeep_regex.append(r'^(bene_dob)$')
        if hmo_val is not None:
            tokeep_regex.append(r'^(hmoind\d{2})$')
            if hmo_months == 'age_year':
                tokeep_regex.append(r'^(bene_dob)$')

        if keep_vars is not None:
            for var in keep_vars:
                tokeep_regex.append(r'^({})$'.format(var))

        tokeep_regex = '|'.join(tokeep_regex)

        tokeep_vars = {}
        for year in self.years:
            if self.parquet_engine == 'pyarrow':
                pf = pq.ParquetFile(self.fpath(self.percent, year, 'bsfab'))
                cols = pf.schema.names
            elif self.parquet_engine == 'fastparquet':
                pf = fp.ParquetFile(self.fpath(self.percent, year, 'bsfab'))
                cols = pf.columns

            tokeep_vars[year] = [x for x in cols if re.search(tokeep_regex, x)]

        # Now perform extraction
        extracted_dfs = []
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
                print(mywrap(msg))

            if self.parquet_engine == 'pyarrow':
                pf = pq.ParquetFile(self.fpath(self.percent, year, 'bsfab'))
                pl = pf.read(
                    columns=tokeep_vars[year],
                    nthreads=min(len(tokeep_vars[year]),
                                 self.parquet_nthreads)).to_pandas().set_index(
                                     'bene_id')
            elif self.parquet_engine == 'fastparquet':
                pf = fp.ParquetFile(self.fpath(self.percent, year, 'bsfab'))
                pl = pf.to_pandas(columns=tokeep_vars[year], index='bene_id')

            nobs = len(pl)

            if gender is not None:
                if (gender.lower() == 'male') | (gender.lower() == 'm'):
                    if pl.sex.dtype.name == 'category':
                        pl.drop(pl[pl['sex'] == '2'].index, inplace=True)
                    elif np.issubdtype(pl.sex.dtype, np.number):
                        pl.drop(pl[pl['sex'] == 2].index, inplace=True)
                elif (gender.lower() == 'female') | (gender.lower() == 'f'):
                    if pl.sex.dtype.name == 'category':
                        pl.drop(pl[pl['sex'] == '1'].index, inplace=True)
                    elif np.issubdtype(pl.sex.dtype, np.number):
                        pl.drop(pl[pl['sex'] == 1].index, inplace=True)

                if 'sex' not in keep_vars:
                    pl.drop('sex', axis=1, inplace=True)

                nobs_dropped[year]['gender'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            if ages is not None:
                pl = pl.loc[pl['age'].isin(ages)]

                pl.drop('age', axis=1, inplace=True)

                nobs_dropped[year]['age'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            if races is not None:
                pl = pl.loc[pl[race_col].isin(races)]

                if race_col not in keep_vars:
                    pl.drop(race_col, axis=1, inplace=True)

                nobs_dropped[year]['race'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            pl.columns = [f'{x}{year}' for x in pl.columns]

            extracted_dfs.append(pl)

        if verbose & (len(extracted_dfs) > 1):
            msg = f"""\
            Merging together beneficiary files
            - years: {self.years}
            - merge type: {join}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(mywrap(msg))

        # @NOTE As long as I'm only looking across years,
        # doing a left join on the last year should be fine
        if len(extracted_dfs) == 1:
            pl = extracted_dfs[0]
        elif len(extracted_dfs) == 2:
            if join == 'default':
                pl = extracted_dfs[0].join(extracted_dfs[1], how='left')
            else:
                pl = extracted_dfs[0].join(extracted_dfs[1], how=join)
        else:
            if join == 'default':
                pl = extracted_dfs[0].join(
                    extracted_dfs[1:-1], how='outer').join(
                        extracted_dfs[-1], how='left')
            else:
                pl = extracted_dfs[0].join(extracted_dfs[1:], how=join)

        pl.index.name = 'bene_id'

        if (((buyin_val is not None) and (buyin_months == 'age_year'))
                or ((hmo_val is not None) and (hmo_months == 'age_year'))):

            # Create month of birth variable
            pl['bene_dob'] = pd.NaT
            for year in self.years:
                pl['bene_dob'] = pl['bene_dob'].combine_first(
                    pl[f'bene_dob{year}'])
                pl.drop(f'bene_dob{year}', axis=1, inplace=True)

            pl['dob_month'] = pl['bene_dob'].dt.month

        if buyin_val is not None:
            if verbose:
                msg = f"""\
                Filtering based on buyin_val
                - values: {buyin_val}
                - filter type: {buyin_months}
                - time elapsed: {(time() - t0) / 60:.2f} minutes
                """
                print(mywrap(msg))

            if buyin_months == 'age_year':

                # Create indicator variable for each year if `buyin ==
                # buyin_val` for the 13 months starting in birthday month of
                # `year` and ending in birthday month of `year + 1`

                for year in self.years[:-1]:
                    # Initialize indicator variable for each year
                    pl[f'buyin_match_{year}'] = False

                    for month in range(1, 13):
                        buyin_cols = []
                        for colname in pl.columns:
                            match = re.search(r'buyin(\d{2})(\d{4})', colname)
                            if match is not None:
                                # Match month
                                m_month = int(match[1])
                                # Match year
                                m_year = int(match[2])
                                if (m_month >= month) & (m_year == year):
                                    buyin_cols.append(colname)
                                elif (m_month <= month) & (m_year == year + 1):
                                    buyin_cols.append(colname)

                        pl.loc[(pl['dob_month'] == month)
                               & (pl[buyin_cols].isin(buyin_val)).all(axis=1),
                               f'buyin_match_{year}'] = True

                    nobs_dropped[year]['buyin'] = (
                        1 - (pl[f'buyin_match_{year}'].sum() / len(pl)))

                regex = re.compile(r'^buyin_match_\d{4}$').search
                buyin_match_cols = [x for x in pl if regex(x)]
                pl = pl.loc[pl[buyin_match_cols].all(axis=1)]

                regex = re.compile(r'^buyin\d{2}\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                cols_todrop.extend(buyin_match_cols)
                pl.drop(cols_todrop, axis=1, inplace=True)

            elif buyin_months == 'all':
                buyin_cols = [x for x in pl if re.search(r'^buyin\d{2}', x)]
                pl = pl.loc[(pl[buyin_cols].isin(buyin_val)).all(axis=1)]

                regex = re.compile(r'^buyin\d{2}\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                pl = pl.drop(cols_todrop, axis=1)

        if hmo_val is not None:
            if verbose:
                msg = f"""\
                Filtering based on hmo_val
                - values: {hmo_val}
                - filter type: {hmo_months}
                - time elapsed: {(time() - t0) / 60:.2f} minutes
                """
                print(mywrap(msg))

            if hmo_months == 'age_year':

                # Create indicator variable for each year if `hmo ==
                # hmo_val` for the 13 months starting in birthday month of
                # `year` and ending in birthday month of `year + 1`

                for year in self.years[:-1]:
                    # Initialize indicator variable for each year
                    pl[f'hmo_match_{year}'] = False

                    for month in range(1, 13):
                        hmo_cols = []
                        for colname in pl.columns:
                            match = re.search(r'hmoind(\d{2})(\d{4})', colname)
                            if match is not None:
                                # Match month
                                m_month = int(match[1])
                                # Match year
                                m_year = int(match[2])
                                if (m_month >= month) & (m_year == year):
                                    hmo_cols.append(colname)
                                elif (m_month <= month) & (m_year == year + 1):
                                    hmo_cols.append(colname)

                        pl.loc[(pl['dob_month'] == month)
                               & (pl[hmo_cols].isin(hmo_val)).all(axis=1),
                               f'hmo_match_{year}'] = True

                    nobs_dropped[year]['hmo'] = (
                        1 - (pl[f'hmo_match_{year}'].sum() / len(pl)))

                regex = re.compile(r'^hmo_match_\d{4}$').search
                hmo_match_cols = [x for x in pl if regex(x)]
                pl = pl.loc[pl[hmo_match_cols].all(axis=1)]

                regex = re.compile(r'^hmoind\d{2}\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                cols_todrop.extend(hmo_match_cols)
                pl.drop(cols_todrop, axis=1, inplace=True)

            elif hmo_months == 'all':
                hmo_cols = [x for x in pl if re.search(r'^hmoind\d{2}', x)]
                pl = pl.loc[(pl[hmo_cols].isin(hmo_val)).all(axis=1)]

                regex = re.compile(r'^hmoind\d{2}\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                pl = pl.drop(cols_todrop, axis=1)

        if (((buyin_val is not None) and (buyin_months == 'age_year'))
                or ((hmo_val is not None) and (hmo_months == 'age_year'))):

            pl.drop('dob_month', axis=1, inplace=True)

            if 'bene_dob' not in keep_vars:
                pl.drop('bene_dob', axis=1, inplace=True)

        # Create single variable across years for any non month-oriented vars
        # Columns that vary by year:
        regex = re.compile(r'(?!_).\d{4}$').search
        year_cols = [x for x in pl if regex(x)]

        # unique names of columns that vary by year:
        year_cols_stub = list(set([x[:-4] for x in year_cols]))

        if year_cols != []:
            pl = pd.wide_to_long(
                pl.reset_index(),
                stubnames=year_cols_stub,
                i='bene_id',
                j='year')

            pl = pl.reset_index('year').drop('year', axis=1)
            pl = pl[~pl.index.duplicated(keep='first')]

        self.nobs_dropped = nobs_dropped
        self.pl = pl

        if verbose:
            msg = f"""\
            Finished cohort retrieval
            - percent sample: {self.percent}
            - years: {self.years}
            - ages: {list(ages)}
            - races: {list(races)}
            - buyin values: {buyin_val}
            - buyin months: {buyin_months}
            - HMO values: {hmo_val}
            - HMO months: {hmo_months}
            - extra variables: {keep_vars}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(mywrap(msg))

    @staticmethod
    def _check_code_types(var):
        """Check type of hcpcs, icd9_dx, icd9_sg codes

        Args:
            var: variable to check types of

        Returns:
            var

        Raises:
            TypeError if wrong type
        """

        # If provided with str or compiled regex, coerce to list
        if type(var) == str:
            var = [var]
        elif isinstance(var, re._pattern_type):
            var = [var]
        elif type(var) == list:
            # Check all elements of list are same type
            if type(var[0]) == str:
                assert all((type(x) is str) for x in var)
            elif isinstance(var[0], re._pattern_type):
                assert all(isinstance(x, re._pattern_type) for x in var)
            else:
                raise TypeError('Codes must be str or compiled regex')
        else:
            raise TypeError('Codes must be str or compiled regex')

        return var

    @staticmethod
    def get_pattern(obj):
        """
        If str, returns str. If compiled regex, returns pattern
        """
        if type(obj) == str:
            return obj
        elif isinstance(obj, re._pattern_type):
            return obj.pattern
        else:
            raise TypeError('Provided non string or regex to get_pattern()')

    def create_rename_dict(
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
        msg = mywrap(msg)

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
                rename_new[self.get_pattern(item)] = ''

        if icd9_dx is not None:
            for item in icd9_dx:
                rename_new[self.get_pattern(item)] = ''

        if icd9_sg is not None:
            for item in icd9_sg:
                rename_new[self.get_pattern(item)] = ''

        # Now fill in rename_new using rename
        msg = 'The values of the rename dictionary must be type list or dict'
        if type(rename.get('hcpcs')) == list:
            for i in range(len(rename['hcpcs'])):
                key = self.get_pattern(hcpcs[i])
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
                key = self.get_pattern(icd9_dx[i])
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
                key = self.get_pattern(icd9_sg[i])
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

    def search_for_codes(
            self,
            data_types,
            hcpcs=None,
            icd9_dx=None,
            icd9_dx_max_cols=None,
            icd9_sg=None,
            keep_vars={},
            collapse_codes=False,
            rename={
                'hcpcs': None,
                'icd9_dx': None,
                'icd9_sg': None},
            convert_ehic=True,
            verbose=False):
        """Search in given claim-level dataset(s) for HCPCS and/or ICD9 codes

        Note: Each code given must be distinct, or collapse_codes must be True

        Args:
            data_types (str or list[str]): carc, carl, ipc, ipr, med, opc, opr
            hcpcs (str, compiled regex, list[str], list[compiled regex]):
                List of HCPCS codes to look for
            icd9_dx (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 diagnosis codes to look for
            icd9_dx_max_cols (int): Max number of ICD9 diagnosis code columns to
                search through
            icd9_sg (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 procedure codes to look for
            keep_vars (dict[data_type: list[str]]): dict of column names to return
            collapse_codes (bool): If True, returns a single column "match";
                else it returns a column for each code provided
            convert_ehic (bool): If True, merges on 'bene_id' for years < 2006

        Returns:
            DataFrame with bene_id and bool columns for each code to search for
        """

        if self.verbose:
            verbose = True

        if verbose:
            t0 = time()
            msg = f"""\
            Starting searching for codes
            - percent sample: {self.percent}
            - years: {self.years}
            - data_types: {data_types}
            """
            print(mywrap(msg))

        if type(data_types) is str:
            data_types = [data_types]

        data_types = set(data_types)

        ok_data_types = ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']
        ok_hcpcs_data_types = ['carl', 'ipr', 'opr']
        ok_dx_data_types = ['carc', 'carl', 'ipc', 'med', 'opc']
        ok_sg_data_types = ['ipc', 'med', 'opc']

        # Instantiate all data types in the keep_vars dict
        for data_type in ok_data_types:
            keep_vars[data_type] = keep_vars.get(data_type, [])

            if type(keep_vars[data_type]) is str:
                keep_vars[data_type] = [keep_vars[data_type]]

        # Check that all data types provided to search through exist
        if not data_types.issubset(ok_data_types):
            invalid_vals = list(data_types.difference(ok_data_types))
            msg = f"""\
            {invalid_vals} does not match any dataset.
            - Allowed data_types: {ok_data_types}
            """
            raise ValueError(mywrap(msg))

        # Check types of codes given, i.e. that all are strings or
        # compiled regexes, and print which codes are searched in which dataset
        if verbose:
            msg = f"""\
            Will check the following codes
            - years: {self.years}
            """
            msg = mywrap(msg)

        all_codes = []
        if hcpcs is not None:
            hcpcs = self._check_code_types(hcpcs)
            all_codes.extend(hcpcs)
            if verbose:
                dts = list(data_types.intersection(ok_hcpcs_data_types))
                msg += mywrap(
                    f"""\
                - HCPCS codes: {hcpcs}
                  in data types: {dts}
                """)

        if icd9_dx is not None:
            icd9_dx = self._check_code_types(icd9_dx)
            all_codes.extend(icd9_dx)
            if verbose:
                dts = list(data_types.intersection(ok_dx_data_types))
                msg += mywrap(
                    f"""\
                - ICD-9 diagnosis codes: {icd9_dx}
                  in data types: {dts}
                """)

        if icd9_sg is not None:
            icd9_sg = self._check_code_types(icd9_sg)
            all_codes.extend(icd9_sg)
            if verbose:
                dts = list(data_types.intersection(ok_sg_data_types))
                msg += mywrap(
                    f"""\
                - ICD-9 procedure codes: {icd9_sg}
                  in data types: {dts}
                """)

        if verbose:
            print(msg)

        all_codes = [self.get_pattern(x) for x in all_codes]
        msg = 'Code patterns given must be unique'
        assert len(all_codes) == len(set(all_codes)), msg

        rename = self.create_rename_dict(
            hcpcs=hcpcs, icd9_dx=icd9_dx, icd9_sg=icd9_sg, rename=rename)

        data = {}
        for data_type in data_types:
            data[data_type] = {}
            for year in self.years:
                if verbose:
                    msg = mywrap(
                        f"""\
                    Starting search for codes
                    - year: {year}
                    - data type: {data_type}
                    """)
                    if data_type in ok_hcpcs_data_types:
                        if hcpcs is not None:
                            msg += mywrap(
                                f"""\
                            - HCPCS codes: {hcpcs}
                            """)
                    if data_type in ok_dx_data_types:
                        if icd9_dx is not None:
                            msg += mywrap(
                                f"""\
                            - ICD-9 diagnosis codes: {icd9_dx}
                            """)
                    if data_type in ok_sg_data_types:
                        if icd9_sg is not None:
                            msg += mywrap(
                                f"""\
                            - ICD-9 procedure codes: {icd9_sg}
                            """)
                    if keep_vars[data_type] != []:
                        msg += mywrap(
                            f"""\
                        - Keeping variables: {keep_vars[data_type]}
                        """)
                    msg += mywrap(
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
            - years: {self.years}
            - data types: {data_types}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(mywrap(msg))

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
                            self.fpath(self.percent, year, 'bsfab'))
                        pl = pf.read(
                            columns=['ehic', 'bene_id'],
                            nthreads=2).to_pandas().set_index('ehic')
                    elif self.parquet_engine == 'fastparquet':
                        pf = fp.ParquetFile(
                            self.fpath(self.percent, year, 'bsfab'))
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
            - years: {self.years}
            - data_types: {data_types}
            - time elapsed: {(time() - t0) / 60:.2f} minutes
            """
            print(mywrap(msg))

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
        all_cols = fp.ParquetFile(self.fpath(self.percent, year,
                                             data_type)).columns
        cols = [x for x in all_cols if regex(x)]

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
            pf = pq.ParquetFile(self.fpath(self.percent, year, data_type))
            itr = (
                pf.read_row_group(
                    i,
                    columns=cols,
                    nthreads=min(len(cols), self.parquet_nthreads)).to_pandas()
                .set_index(pl_id_col) for i in range(pf.num_row_groups))
        elif self.parquet_engine == 'fastparquet':
            pf = fp.ParquetFile(self.fpath(self.percent, year, data_type))
            itr = pf.iter_row_groups(columns=cols, index=pl_id_col)

        for cl in itr:
            if pl_ids_to_filter is not None:
                index_name = cl.index.name
                cl = cl.join(pd.DataFrame(index=pl_ids_to_filter), how='inner')
                cl.index.name = index_name

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

                    cl.drop(hcpcs_cols, axis=1, inplace=True)

                if icd9_dx:
                    for code in icd9_dx:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_dx_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_dx_cols] == code
                                   ).any(axis=1), 'match'] = True

                    cl.drop(icd9_dx_cols, axis=1, inplace=True)

                if icd9_sg:
                    for code in icd9_sg:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_sg_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_sg_cols] == code
                                   ).any(axis=1), 'match'] = True

                    cl.drop(icd9_sg_cols, axis=1, inplace=True)

                # Keep all rows; not just matches
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

                    cl.drop(hcpcs_cols, axis=1, inplace=True)

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

                    cl.drop(icd9_dx_cols, axis=1, inplace=True)

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

                    cl.drop(icd9_sg_cols, axis=1, inplace=True)

                cl['match'] = (cl[all_created_cols] == True).any(axis=1)

                # Rename columns according to `rename` dictionary
                cl = cl.rename(index=str, columns=rename)

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

    def search_for_codes_pl(
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
