#! /usr/bin/env python3
"""Main module."""

import re
import pandas as pd
import fastparquet as fp


def fpath(percent: str, year: int, data_type: str, dta: bool=False,
          med_dta: str='/disk/aging/medicare/data',
          med_pq: str='/homes/nber/barronk/agebulk1/raw'):
    """Generate path to Medicare files

    Args:
        percent: percent sample to convert
        year: year of data to convert
        data_type:
            - bsfab Beneficiary Summary File, Base segment
            - bsfcc Beneficiary Summary File, Chronic Conditions segment
            - bsfcu Beneficiary Summary File, Cost & Use segment
            - bsfd  Beneficiary Summary File, National Death Index segment
            - carc  Carrier File, Claims segment
            - carl  Carrier File, Line segment
            - den   Denominator File
            - dmec  Durable Medical Equipment File, Claims segment
            - dmel  Durable Medical Equipment File, Line segment
            - hhac  Home Health Agency File, Claims segment
            - hhar  Home Health Agency File, Revenue Center segment
            - hosc  Hospice File, Claims segment
            - hosr  Hospice File, Revenue Center segment
            - ipc   Inpatient File, Claims segment
            - ipr   Inpatient File, Revenue Center segment
            - med   MedPAR File
            - opc   Outpatient File, Claims segment
            - opr   Outpatient File, Revenue Center segment
            - snfc  Skilled Nursing Facility File, Claims segment
            - snfr  Skilled Nursing Facility File, Revenue Center segment
            - xw    `ehic` - `bene_id` crosswalk files
        dta: Returns Stata file path
        med_dta: top of tree for medicare stata files
        med_pq: top of tree for medicare parquet files
    Returns:
        string with file path.
    Raises:
        NameError if data_type is not one of the above
    """

    from os.path import isfile

    if type(data_type) != str:
        raise TypeError('data_type must be str')

    allowed_data_types = [
        'bsfab', 'bsfcc', 'bsfcu', 'bsfd', 'carc', 'carl', 'den', 'dmec',
        'dmel', 'hhac', 'hhar', 'hosc', 'hosr', 'ipc', 'ipr', 'med', 'opc',
        'opr', 'snfc', 'snfr', 'xw']
    if data_type not in allowed_data_types:
        raise ValueError(f'data_type must be one of:\n{allowed_data_types}')

    if data_type == 'bsfab':
        dta_path = f'{med_dta}/{percent}pct/bsf/{year}/1/bsfab{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/bsf/bsfab{year}.parquet'
    elif data_type == 'bsfcc':
        dta_path = f'{med_dta}/{percent}pct/bsf/{year}/1/bsfcc{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/bsf/bsfcc{year}.parquet'
    elif data_type == 'bsfcu':
        dta_path = f'{med_dta}/{percent}pct/bsf/{year}/1/bsfcu{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/bsf/bsfcu{year}.parquet'
    elif data_type == 'bsfd':
        dta_path = f'{med_dta}/{percent}pct/bsf/{year}/1/bsfd{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/bsf/bsfd{year}.parquet'

    elif data_type == 'carc':
        if year >= 2002:
            dta_path = f'{med_dta}/{percent}pct/car/{year}/carc{year}.dta'
        else:
            dta_path = f'{med_dta}/{percent}pct/car/{year}/car{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/car/carc{year}.parquet'
    elif data_type == 'carl':
        assert year >= 2002
        dta_path = f'{med_dta}/{percent}pct/car/{year}/carl{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/car/carl{year}.parquet'

    elif data_type == 'den':
        dta_path = f'{med_dta}/{percent}pct/den/{year}/den{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/den/den{year}.parquet'

    elif data_type == 'dmec':
        dta_path = f'{med_dta}/{percent}pct/dme/{year}/dmec{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/dme/dmec{year}.parquet'
    elif data_type == 'dmel':
        dta_path = f'{med_dta}/{percent}pct/dme/{year}/dmel{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/dme/dmel{year}.parquet'

    elif data_type == 'hhac':
        dta_path = f'{med_dta}/{percent}pct/hha/{year}/hhac{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/hha/hhac{year}.parquet'
    elif data_type == 'hhar':
        dta_path = f'{med_dta}/{percent}pct/hha/{year}/hhar{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/hha/hhar{year}.parquet'

    elif data_type == 'hosc':
        dta_path = f'{med_dta}/{percent}pct/hos/{year}/hosc{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/hos/hosc{year}.parquet'
    elif data_type == 'hosr':
        dta_path = f'{med_dta}/{percent}pct/hos/{year}/hosr{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/hos/hosr{year}.parquet'

    elif data_type == 'ipc':
        if year >= 2002:
            dta_path = f'{med_dta}/{percent}pct/ip/{year}/ipc{year}.dta'
        else:
            dta_path = f'{med_dta}/{percent}pct/ip/{year}/ip{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/ip/ipc{year}.parquet'
    elif data_type == 'ipr':
        assert year >= 2002
        dta_path = f'{med_dta}/{percent}pct/ip/{year}/ipr{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/ip/ipr{year}.parquet'

    elif data_type == 'med':
        dta_path = f'{med_dta}/{percent}pct/med/{year}/med{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/med/med{year}.parquet'

    elif data_type == 'opc':
        dta_path = f'{med_dta}/{percent}pct/op/{year}/opc{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/op/opc{year}.parquet'
    elif data_type == 'opr':
        dta_path = f'{med_dta}/{percent}pct/op/{year}/opr{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/op/opr{year}.parquet'

    elif data_type == 'snfc':
        dta_path = f'{med_dta}/{percent}pct/snf/{year}/snfc{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/snf/snfc{year}.parquet'
    elif data_type == 'snfr':
        dta_path = f'{med_dta}/{percent}pct/snf/{year}/snfr{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/snf/snfr{year}.parquet'

    elif data_type == 'xw':
        dta_path = f'{med_dta}/{percent}pct/xw/{year}/ehicbenex_one{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/xw/ehicbenex_one{year}.parquet'

    if dta:
        if isfile(dta_path):
            return dta_path
        else:
            raise FileNotFoundError
    else:
        return pq_path


class MedicareDF(object):
    """A class to organize Medicare operations"""

    def __init__(self, percent, years, verbose=False):
        """Return a MedicareDF object

        Attributes:
            percent (str): percent sample of data to use
            years (list[int]): years of data to use
        """

        allowed_pcts = ['0001', '01', '05', '20', '100']

        if type(percent) != str:
            raise TypeError('percent must be str')
        elif percent not in allowed_pcts:
            msg = f'percent must be one of: {allowed_pcts}'
            raise ValueError(msg)
        else:
            self.percent = percent

        if type(years) == int:
            years = [years]
        else:
            years = years

        assert min(years) >= 2001
        assert max(years) <= 2015

        self.years = years
        self.verbose = verbose

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

        cols = fp.ParquetFile(fpath(self.percent, year, data_type)).columns
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
            join_across_years='default',
            keep_vars=[],
            verbose=False):
        """Get cohort in standardized way

        Merges in such a way that age has to be within `ages` in any such year

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
            join_across_years (str): method for joining across years
                Default is "outer" join for all years up to N-1, "left" for N
                Otherwise must be "left", "inner", "outer", "right"
            keep_vars (list[str]): Variable names to keep in final output

        Returns:
            Adds DataFrame of extracted cohort to instance
        """

        import numpy as np

        if self.verbose:
            verbose = True

        if verbose:
            from time import time
            t0 = time()
            msg = 'Starting cohort retrieval\n'
            msg += f'\t- percent sample: {self.percent}\n'
            print(msg)

        if len(self.years) == 1:
            if buyin_months == 'age_year':
                msg = "buyin_months can't be 'age_year' when one year is given"
                raise ValueError(msg)
        if type(ages) == int:
            ages = [ages]
        if type(races) == str:
            races = [races]

        if type(buyin_val) == str:
            buyin_val = [buyin_val]

        if buyin_val is not None:
            allowed_buyin_months = ['all', 'age_year']
            if buyin_months not in allowed_buyin_months:
                msg = f'buyin_months must be one of: {allowed_buyin_months}'
                raise ValueError(msg)

        if type(hmo_val) == str:
            hmo_val = [hmo_val]

        if hmo_val is not None:
            allowed_hmo_months = ['all', 'age_year']
            if hmo_months not in allowed_hmo_months:
                msg = f'hmo_months must be one of: {allowed_hmo_months}'
                raise ValueError(msg)

        allowed_join_across_years = [
            'default', 'left', 'inner', 'outer', 'right']
        if join_across_years not in allowed_join_across_years:
            msg = 'join_across_years must be one of:'
            msg += f' {allowed_join_across_years}'
            raise ValueError(msg)

        if type(keep_vars) == str:
            keep_vars = [keep_vars]

        # Get list of variables to import for each year
        tokeep_regex = []
        tokeep_regex.extend([r'^(ehic)$', r'^(bene_id)$'])
        if gender is not None:
            tokeep_regex.append(r'^(sex)$')
        if ages is not None:
            tokeep_regex.append(r'^(age)$')
        if races is not None:
            if rti_race:
                tokeep_regex.append(r'^(rti_race_cd)$')
            else:
                tokeep_regex.append(r'^(race)$')
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
                tokeep_regex.append(rf'^({var})$')

        tokeep_regex = '|'.join(tokeep_regex)

        tokeep_vars = {}
        for year in self.years:
            cols = fp.ParquetFile(fpath(self.percent, year, 'bsfab')).columns
            tokeep_vars[year] = [x for x in cols if re.search(tokeep_regex, x)]

        # Now perform extraction
        extracted_dfs = []
        nobs_dropped = {}

        # Do filtering for all vars that are
        # checkable within a single year's data
        for year in self.years:
            if verbose:
                msg = 'Importing bsfab file\n'
                msg += f'\t- year: {year}\n'
                msg += f'\t- columns: {tokeep_vars[year]}\n'
                msg += f'\t- time elapsed: {(time() - t0) / 60:.2f} minutes\n'
                print(msg)

            pf = fp.ParquetFile(fpath(self.percent, year, 'bsfab'))
            pl = pf.to_pandas(columns=tokeep_vars[year], index='bene_id')
            nobs = len(pl)
            nobs_dropped[year] = {}

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

                if 'age' not in keep_vars:
                    pl.drop('age', axis=1, inplace=True)

                nobs_dropped[year]['age'] = 1 - (len(pl) / nobs)
                nobs = len(pl)

            pl.columns = [f'{x}_{year}' for x in pl.columns]

            extracted_dfs.append(pl)

        if verbose:
            msg = 'Merging together beneficiary files\n'
            msg += f'\t- years: {self.years}\n'
            msg += f'\t- merge type: {join_across_years}\n'
            msg += f'\t- time elapsed: {(time() - t0) / 60:.2f} minutes\n'
            print(msg)

        # @NOTE As long as I'm only looking across years,
        # doing a left join on the last year should be fine
        if len(extracted_dfs) == 1:
            pl = extracted_dfs[0]
        elif len(extracted_dfs) == 2:
            if join_across_years == 'default':
                pl = extracted_dfs[0].join(extracted_dfs[1], how='left')
            else:
                pl = extracted_dfs[0].join(
                    extracted_dfs[1], how=join_across_years)
        else:
            if join_across_years == 'default':
                pl = extracted_dfs[0].join(
                    extracted_dfs[1:-1], how='outer').join(
                        extracted_dfs[-1], how='left')
            else:
                pl = extracted_dfs[0].join(
                    extracted_dfs[1:], how=join_across_years)

        # Create single variable across years for any non month-oriented
        # variables (i.e. buyin and hmo status)

        # columns that don't vary by month:
        year_cols = [x for x in pl if not re.search(r'\d{2}_\d{4}$', x)]

        # unique names of columns that don't vary by month:
        year_cols_stub = list(set([x[:-5] for x in year_cols]))

        dtypes = dict(pl.dtypes)
        for col in year_cols_stub:
            # Generate stub column with same dtype as col
            dtype = dtypes[f'{col}_{min(self.years)}']
            pl[col] = np.NaN
            pl[col] = pl[col].astype(dtype)

            for year in self.years:
                pl[col] = pl[col].combine_first(pl[f'{col}_{year}'])
                pl.drop(f'{col}_{year}', axis=1, inplace=True)

            pl[col] = pl[col].astype(dtype)

        if (((buyin_val is not None) and (buyin_months == 'age_year'))
                or ((hmo_val is not None) and (hmo_months == 'age_year'))):

            pl['dob_month'] = pl['bene_dob'].dt.month

        if buyin_val is not None:
            if verbose:
                msg = 'Filtering based on buyin_val\n'
                msg += f'\t- values: {buyin_val}\n'
                msg += f'\t- filter type: {buyin_months}\n'
                msg += f'\t- time elapsed: {(time() - t0) / 60:.2f} minutes\n'
                print(msg)

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
                            match = re.search(r'buyin(\d{2})_(\d{4})', colname)
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

                regex = re.compile(r'^buyin\d{2}_\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                cols_todrop.extend(buyin_match_cols)
                pl.drop(cols_todrop, axis=1, inplace=True)

            elif buyin_months == 'all':
                buyin_cols = [x for x in pl if re.search(r'^buyin\d{2}', x)]
                pl = pl.loc[(pl[buyin_cols].isin(buyin_val)).all(axis=1)]

                regex = re.compile(r'^buyin\d{2}_\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                pl = pl.drop(cols_todrop, axis=1)

        if hmo_val is not None:
            if verbose:
                msg = 'Filtering based on hmo_val\n'
                msg += f'\t- values: {hmo_val}\n'
                msg += f'\t- filter type: {hmo_months}\n'
                msg += f'\t- time elapsed: {(time() - t0) / 60:.2f} minutes\n'
                print(msg)

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
                            match = re.search(r'hmoind(\d{2})_(\d{4})', colname)
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

                regex = re.compile(r'^hmoind\d{2}_\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                cols_todrop.extend(hmo_match_cols)
                pl.drop(cols_todrop, axis=1, inplace=True)

            elif hmo_months == 'all':
                hmo_cols = [x for x in pl if re.search(r'^buyin\d{2}', x)]
                pl = pl.loc[(pl[hmo_cols].isin(hmo_val)).all(axis=1)]

                regex = re.compile(r'^buyin\d{2}_\d{4}$').search
                cols_todrop = [x for x in pl if regex(x)]
                pl = pl.drop(cols_todrop, axis=1)

        if (((buyin_val is not None) and (buyin_months == 'age_year'))
                or ((hmo_val is not None) and (hmo_months == 'age_year'))):

            pl.drop('dob_month', axis=1, inplace=True)

            if 'bene_dob' not in keep_vars:
                pl.drop('bene_dob', axis=1, inplace=True)

        self.nobs_dropped = nobs_dropped
        self.pl = pl

    @staticmethod
    def _check_code_types(var):
        """Check type of hcpcs, icd9_diag, icd9_proc codes

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

    def search_for_codes(
            self,
            year,
            data_type,
            hcpcs=None,
            icd9_diag=None,
            icd9_proc=None,
            keep_vars=[],
            collapse_codes=False):
        """Search in given claim-level dataset for HCPCS/ICD9 codes
        NOTE: Will want to remove year?

        Note: Each code given must be distinct, or collapse_codes must be True

        Args:
            year (int): year of data to search
            data_type (str): One of carc, carl, ipc, ipr, med, opc, opr
            hcpcs (str, compiled regex, list[str], list[compiled regex]):
                List of HCPCS codes to look for
            icd9_diag (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 diagnosis codes to look for
            icd9_proc (str, compiled regex, list[str], list[compiled regex]):
                List of ICD-9 procedure codes to look for
            keep_vars (list[str]): list of column names to return
            collapse_codes (bool): If True, returns a single column "match";
                else it returns a column for each code provided

        Returns:
            DataFrame with bene_id and bool columns for each code to search for
        """

        if data_type not in ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']:
            msg = 'data_type provided that does not match any dataset'
            raise ValueError(msg)

        if hcpcs is not None:
            # If variable is not in data_type file, raise error
            if data_type in ['carc', 'ipc', 'med', 'opc']:
                msg = 'data_type was supplied that does not have HCPCS columns'
                raise ValueError(msg)

            hcpcs = self._check_code_types(hcpcs)

        if icd9_diag is not None:
            # If variable is not in data_type file, raise error
            if data_type in ['ipr', 'opr']:
                msg = 'data_type was supplied that does not have columns'
                msg += ' for ICD-9 diagnosis codes'
                raise ValueError(msg)

            icd9_diag = self._check_code_types(icd9_diag)

        if icd9_proc is not None:
            # If variable is not in data_type file, raise error
            if data_type in ['carc', 'carl', 'ipr', 'opr']:
                msg = 'data_type was supplied that does not have columns'
                msg += ' for ICD-9 procedure codes'
                raise ValueError(msg)

            icd9_proc = self._check_code_types(icd9_proc)

        if type(collapse_codes) != bool:
            raise TypeError('collapse_codes must be boolean')

        try:
            bene_ids_to_filter = self.pl.index
        except AttributeError:
            bene_ids_to_filter = None

        pf = fp.ParquetFile(fpath(self.percent, year, data_type))

        # Determine which variables to extract
        regex_string = []
        if data_type == 'med':
            cl_id_regex = r'^medparid$'
            regex_string.append(cl_id_regex)
        else:
            cl_id_regex = r'^clm_id$|^claimindex$'
            regex_string.append(cl_id_regex)

        regex_string.append(r'^ehic$')

        if hcpcs is not None:
            hcpcs_regex = r'^hcpcs_cd$'
            regex_string.append(hcpcs_regex)

        if icd9_diag is not None:
            if data_type == 'carl':
                icd9_diag_regex = r'icd_dgns_cd\d*$'
            elif data_type == 'med':
                icd9_diag_regex = r'^dgnscd\d+$$'
            else:
                icd9_diag_regex = r'^icd_dgns_cd\d+$'
            regex_string.append(icd9_diag_regex)

        if icd9_proc is not None:
            icd9_proc_regex = r'^icd_prcdr_cd\d+$'
            regex_string.append(icd9_proc_regex)

        for var in keep_vars:
            regex_string.append(rf'^{var}$')

        regex_string = '|'.join(regex_string)
        regex = re.compile(regex_string).search
        cols = [x for x in pf.columns if regex(x)]

        cl_id_col = [x for x in cols if re.search(cl_id_regex, x)]
        if hcpcs is not None:
            hcpcs_cols = [x for x in cols if re.search(hcpcs_regex, x)]
        else:
            hcpcs_cols = None

        if icd9_diag is not None:
            icd9_diag_cols = [x for x in cols if re.search(icd9_diag_regex, x)]
        else:
            icd9_diag_cols = None

        if icd9_proc is not None:
            icd9_proc_cols = [x for x in cols if re.search(icd9_proc_regex, x)]
        else:
            icd9_proc_cols = None

        # This holds the df's from each iteration over the claim-level dataset
        all_cl = []

        # cl = pf.to_pandas(columns=cols, index='bene_id')
        for cl in pf.iter_row_groups(columns=cols, index='bene_id'):
            if bene_ids_to_filter is not None:
                cl = cl.join(
                    pd.DataFrame(index=bene_ids_to_filter), how='inner')

            if cl.index.name == 'bene_id':
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

                if icd9_diag:
                    for code in icd9_diag:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_diag_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_diag_cols] == code
                                    ).any(axis=1), 'match'] = True

                    cl.drop(icd9_diag_cols, axis=1, inplace=True)

                if icd9_proc:
                    for code in icd9_proc:
                        if isinstance(code, re._pattern_type):
                            cl.loc[cl[icd9_proc_cols].apply(
                                lambda col: col.str.contains(code)).any(
                                    axis=1), 'match'] = True
                        else:
                            cl.loc[(cl[icd9_proc_cols] == code
                                    ).any(axis=1), 'match'] = True

                    cl.drop(icd9_proc_cols, axis=1, inplace=True)

                # Unsure whether to keep only true matches
                # cl = cl.loc[cl['match']]
                all_cl.append(cl)

            else:
                if hcpcs:
                    for code in hcpcs:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[hcpcs_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True

                        else:
                            cl[code] = False
                            idx = cl.index[(cl[hcpcs_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True

                    cl.drop(hcpcs_cols, axis=1, inplace=True)

                if icd9_diag:
                    for code in icd9_diag:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[icd9_diag_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True

                        else:
                            cl[code] = False
                            idx = cl.index[(
                                cl[icd9_diag_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True

                    cl.drop(icd9_diag_cols, axis=1, inplace=True)

                if icd9_proc:
                    for code in icd9_proc:
                        if isinstance(code, re._pattern_type):
                            cl[code.pattern] = False
                            idx = cl.index[cl[icd9_proc_cols].apply(
                                lambda col: col.str.contains(code)).any(axis=1)]
                            cl.loc[idx, code.pattern] = True

                        else:
                            cl[code] = False
                            idx = cl.index[(
                                cl[icd9_proc_cols] == code).any(axis=1)]
                            cl.loc[idx, code] = True

                    cl.drop(icd9_proc_cols, axis=1, inplace=True)

                all_cl.append(cl)

        cl = pd.concat(all_cl, axis=0)
        # Merge back onto bene_ids_to_filter so that claim-level df
        # has same index values as person-level df
        cl = cl.reset_index().merge(
            pd.DataFrame(index=bene_ids_to_filter),
            how='outer',
            left_on='bene_id',
            right_index=True).set_index('bene_id')

        self.cl = cl

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

            if icd9_diag:
                for code in icd9_diag:
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

            if icd9_proc:
                for code in icd9_proc:
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

        self.pl = pl


def pq_vars(ParquetFile):
    import re
    from natsort import natsorted

    varnames = str(ParquetFile.schema).split('\n')
    varnames = [m[1] for m in (re.search(r'(\w+):', x) for x in varnames) if m]
    varnames = natsorted(varnames)
    return varnames
