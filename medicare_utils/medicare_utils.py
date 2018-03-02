#! /usr/bin/env python3

"""Main module."""

import re
import pandas as pd
import fastparquet as fp


def fpath(percent: str, year: int, data_type: str, dta: bool=False):
    """Generate path to Medicare files

    Args:
        percent: percent sample to convert
        year: year of data to convert
        dta: Returns Stata file path
        data_type:
            - carc
            - carl
            - den
            - ipc
            - ipr
            - med
            - op
            - opc
            - opr
            - bsfab
            - bsfcc
            - bsfcu
            - bsfd
    Returns:
        string with file path.
    Raises:
        NameError if data_type is not one of the above
    """

    med_dta = '/disk/aging/medicare/data'
    med_pq = '/homes/nber/barronk/agebulk1/raw'

    if data_type == 'carc':
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

    elif data_type == 'op':
        raise Exception('Haven\'t added support yet for older op files')

    elif data_type == 'opc':
        dta_path = f'{med_dta}/{percent}pct/op/{year}/opc{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/op/opc{year}.parquet'

    elif data_type == 'opr':
        dta_path = f'{med_dta}/{percent}pct/op/{year}/opr{year}.dta'
        pq_path = f'{med_pq}/pq/{percent}pct/op/opr{year}.parquet'

    elif data_type == 'bsfab':
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

    else:
        raise NameError('Unknown data type')

    if dta:
        return dta_path
    else:
        return pq_path


# pct = '01'
# years = 2009
# gender = 'female'
# ages = range(65, 70)
# races = None
# buyin_val = '3'
# buyin_months = 'all'
# join_across_years = 'default'
# vars_to_keep = []
# @TODO add option to remove people who died during year
# @TODO add verbose option
def get_cohort(pct, years, gender=None, ages=None, races=None,
               rti_race=False, buyin_val=None, buyin_months=None,
               join_across_years='default', vars_to_keep=[]):
    """Get cohort in standardized way

    Merges in such a way that age has to be within `ages` in any such year

    Args:
        pct (str): percent sample of data to use
        years (range, list[int], int):
            years of beneficiary data to get cohort from
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
        vars_to_keep (list[str]): Variable names to keep in final output

    Returns:
        DataFrame of extracted cohort
    """

    import numpy as np

    if type(years) == int:
        if buyin_months == 'age_year':
            raise ValueError("Year can't be int when buyin_months is age_year")
        years = [years]
    if type(ages) == int:
        ages = [ages]
    if type(races) == str:
        races = [races]
    if type(buyin_val) == str:
        buyin_val = [buyin_val]
    if type(vars_to_keep) == str:
        vars_to_keep = [vars_to_keep]

    # Get list of variables to import for each year
    tokeep_regex = []
    tokeep_regex.extend([r'^(ehic)$', r'^(bene_id)$'])
    if gender is not None:
        tokeep_regex.append(r'^(sex)$')
    if ages is not None:
        tokeep_regex.append(r'^(age)$')
    if buyin_val is not None:
        tokeep_regex.append(r'^(buyin\d{2})$')
        if buyin_months == 'age_year':
            tokeep_regex.append(r'^(bene_dob)$')
    if races is not None:
        if rti_race:
            tokeep_regex.append(r'^(rti_race_cd)$')
        else:
            tokeep_regex.append(r'^(race)$')

    if vars_to_keep is not None:
        for var in vars_to_keep:
            tokeep_regex.append(rf'^({var})$')

    tokeep_regex = '|'.join(tokeep_regex)

    tokeep_vars = {}
    for year in years:
        cols = fp.ParquetFile(fpath(pct, year, 'bsfab')).columns
        tokeep_vars[year] = [x for x in cols if re.search(tokeep_regex, x)]

    # Now perform extraction
    extracted_dfs = []
    nobs_dropped = {}

    # Do filtering for all vars that are checkable within a single year's data
    for year in years:
        pf = fp.ParquetFile(fpath(pct, year, 'bsfab'))
        demo = pf.to_pandas(columns=tokeep_vars[year], index='bene_id')
        nobs = len(demo)
        nobs_dropped[year] = {}

        if gender is not None:
            if (gender.lower() == 'male') | (gender.lower() == 'm'):
                if demo.sex.dtype.name == 'category':
                    demo.drop(demo[demo['sex'] == '2'].index, inplace=True)
                elif np.issubdtype(demo.sex.dtype, np.number):
                    demo.drop(demo[demo['sex'] == 2].index, inplace=True)
            elif (gender.lower() == 'female') | (gender.lower() == 'f'):
                if demo.sex.dtype.name == 'category':
                    demo.drop(demo[demo['sex'] == '1'].index, inplace=True)
                elif np.issubdtype(demo.sex.dtype, np.number):
                    demo.drop(demo[demo['sex'] == 1].index, inplace=True)

            if 'sex' not in vars_to_keep:
                demo.drop('sex', axis=1, inplace=True)

            nobs_dropped[year]['gender'] = 1 - (len(demo) / nobs)
            nobs = len(demo)

        if ages is not None:
            demo = demo.loc[demo['age'].isin(ages)]

            if 'age' not in vars_to_keep:
                demo.drop('age', axis=1, inplace=True)

            nobs_dropped[year]['age'] = 1 - (len(demo) / nobs)
            nobs = len(demo)

        demo.columns = [f'{x}_{year}' for x in demo.columns]

        extracted_dfs.append(demo)

    # @NOTE As long as I'm only looking across years, doing a left join on the
    # last year should be fine
    if len(extracted_dfs) == 1:
        demo = extracted_dfs[0]

    elif len(extracted_dfs) == 2:
        if join_across_years == 'default':
            demo = extracted_dfs[0].join(extracted_dfs[1], how='left')
        else:
            demo = extracted_dfs[0].join(
                extracted_dfs[1], how=join_across_years)

    else:
        if join_across_years == 'default':
            demo = extracted_dfs[0].join(extracted_dfs[1:-1], how='outer').join(
                extracted_dfs[-1], how='left')
        else:
            demo = extracted_dfs[0].join(
                extracted_dfs[1:], how=join_across_years)

    # Create single variable across years for any non buyin_variables
    # TODO Make this general for all variables
    dob_cols = [x for x in demo if re.search(r'^bene_dob', x)]
    demo['bene_dob'] = pd.NaT
    for col in dob_cols:
        demo['bene_dob'] = demo['bene_dob'].combine_first(demo[col])
    demo.drop(dob_cols, axis=1, inplace=True)

    if buyin_val is not None:
        if buyin_months == 'age_year':

            demo['dob_month'] = demo['bene_dob'].dt.month

            # Create indicator variable for each year if `buyin == buyin_val`
            # for the 13 months starting in birthday month of `year` and ending
            # in birthday month of `year + 1`

            for year in years[:-1]:
                # Initialize indicator variable for each year
                demo[f'buyin_match_{year}'] = False

                for month in range(1, 13):
                    buyin_cols = []
                    for colname in demo:
                        match = re.search(r'buyin(\d{2})_(\d{4})', colname)
                        if match is not None:
                            mt_month = int(match[1])
                            mt_year = int(match[2])
                            if (mt_month >= month) & (mt_year == year):
                                buyin_cols.append(colname)
                            elif (mt_month <= month) & (mt_year == year + 1):
                                buyin_cols.append(colname)

                    demo.loc[(demo['dob_month'] == month)
                             & (demo[buyin_cols].isin(buyin_val)).all(axis=1),
                             f'buyin_match_{year}'] = True

                nobs_dropped[year]['buyin'] = (
                    1 - (demo[f'buyin_match_{year}'].sum() / len(demo)))

            regex = re.compile(r'^buyin_match_\d{4}$').search
            buyin_match_cols = [x for x in demo if regex(x)]
            demo = demo.loc[demo[buyin_match_cols].all(axis=1)]

            regex = re.compile(r'^buyin\d{2}_\d{4}$').search
            cols_todrop = [x for x in demo if regex(x)]
            cols_todrop.append('dob_month')
            cols_todrop.extend(buyin_match_cols)
            demo.drop(cols_todrop, axis=1, inplace=True)

        elif buyin_months == 'all':
            buyin_cols = [x for x in demo if re.search(r'^buyin\d{2}', x)]
            demo = demo.loc[(demo[buyin_cols].isin(buyin_val)).all(axis=1)]

            regex = re.compile(r'^buyin\d{2}_\d{4}$').search
            cols_todrop = [x for x in demo if regex(x)]
            demo = demo.drop(cols_todrop, axis=1)

        else:
            msg = 'Have coded only age_year for buyin_months'
            raise NotImplementedError(msg)

    return demo


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


def _search_codes(df, demo, cols, codes, collapse):
    """Search through file for codes

    Note: Modifies in place

    Args:
        df (pd.DataFrame): file with codes to search through
        demo (pd.DataFrame): demographic file to track who has given code
        cols (list[str]): column names to search over
        codes (list[str] or list[re._pattern_type]): codes to match against
        collapse (bool): whether to return code-matches individually or not
    """
    if collapse:
        demo['match'] = False

        if isinstance(codes[0], re._pattern_type):
            idxs = []
            for code in codes:
                idxs.append(
                    df.index[df[cols].apply(
                        lambda x: x.str.contains(code)).any(axis=1)])

            if len(idxs) == 1:
                idx = idxs[0].unique()
            elif len(idxs) == 2:
                idx = idxs[0].append(idxs[1]).unique()
            else:
                idx = idxs[0].append(idxs[1:]).unique()
            demo.loc[idx, 'match'] = True

        else:
            idx = df.index[(df[cols].isin(codes)).any(axis=1)]
            demo.loc[idx, 'match'] = True

    else:
        for code in codes:
            if isinstance(code, re._pattern_type):
                demo[code.pattern] = False
                idx = df.index[df[cols].apply(
                    lambda x: x.str.contains(code)).any(axis=1)]
                demo.loc[idx, code.pattern] = True
            else:
                demo[code] = False
                idx = df.index[(df[cols] == code).any(axis=1)]
                demo.loc[idx, code] = True

    return demo


def search_for_codes(pct, year, data_type, bene_ids_to_filter=None,
                     hcpcs=None, icd9_diag=None,
                     icd9_proc=None, collapse_codes=False):
    """Search in given dataset for HCPCS/ICD9 codes

    Note: Each code given must be distinct, or collapse_codes must be True

    Args:
        pct (str): percent sample of data to use
        year (int): year of data to search
        data_type (str): One of carc, carl, ipc, ipr, med, opc, opr
        bene_ids_to_filter (Index, list): List of bene_ids to search over
        hcpcs (str, compiled regex, list[str], list[compiled regex]):
            List of HCPCS codes to look for
        icd9_diag (str, compiled regex, list[str], list[compiled regex]):
            List of ICD-9 diagnosis codes to look for
        icd9_proc (str, compiled regex, list[str], list[compiled regex]):
            List of ICD-9 procedure codes to look for
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

        hcpcs = _check_code_types(hcpcs)

    if icd9_diag is not None:
        # If variable is not in data_type file, raise error
        if data_type in ['ipr', 'opr']:
            msg = 'data_type was supplied that does not have columns for ICD-9'
            msg += 'diagnosis codes'
            raise ValueError(msg)

        icd9_diag = _check_code_types(icd9_diag)

    if icd9_proc is not None:
        # If variable is not in data_type file, raise error
        if data_type in ['carc', 'carl', 'ipr', 'opr']:
            msg = 'data_type was supplied that does not have columns for ICD-9'
            msg += 'procedure codes'
            raise ValueError(msg)

        icd9_proc = _check_code_types(icd9_proc)

    if type(pct) != str:
        raise TypeError('pct must be string')
    if type(year) != int:
        raise TypeError('year must be int')

    pf = fp.ParquetFile(fpath(pct, year, data_type))

    # Determine which variables to extract
    regex_string = []
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

    regex_string = '|'.join(regex_string)
    regex = re.compile(regex_string).search
    cols = [x for x in pf.columns if regex(x)]

    all_demo = []
    for df in pf.iter_row_groups(columns=cols, index='bene_id'):

        if bene_ids_to_filter is not None:
            demo = pd.DataFrame(index=bene_ids_to_filter)
            df = df.join(demo, how='inner')
        else:
            demo = pd.DataFrame(index=df.index.unique())

        if hcpcs is not None:
            hcpcs_cols = [x for x in df if re.search(hcpcs_regex, x)]
            demo = _search_codes(df, demo, hcpcs_cols, hcpcs, collapse_codes)

        if icd9_diag is not None:
            icd9_diag_cols = [x for x in df if re.search(icd9_diag_regex, x)]
            demo = _search_codes(df, demo, icd9_diag_cols,
                                 icd9_diag, collapse_codes)

        if icd9_proc is not None:
            icd9_proc_cols = [x for x in df if re.search(icd9_proc_regex, x)]
            demo = _search_codes(df, demo, icd9_proc_cols,
                                 icd9_proc, collapse_codes)

        all_demo.append(demo)

    if len(all_demo) == 1:
        demo = all_demo[0]
    else:
        demo_concat = pd.concat(all_demo, axis=1, join='outer')

        if not collapse_codes:
            demo = pd.DataFrame(index=demo_concat.index)
            if hcpcs is not None:
                for code in hcpcs:
                    demo[code] = demo_concat[code].max(axis=1)

            if icd9_diag is not None:
                for code in icd9_diag:
                    demo[code] = demo_concat[code].max(axis=1)

            if icd9_proc is not None:
                for code in icd9_proc:
                    demo[code] = demo_concat[code].max(axis=1)
        else:
            demo = demo_concat['match'].max(axis=1).to_frame('match')

    return demo


def pq_vars(ParquetFile):
    import re
    from natsort import natsorted

    varnames = str(ParquetFile.schema).split('\n')
    varnames = [m[1] for m in (re.search(r'(\w+):', x) for x in varnames) if m]
    varnames = natsorted(varnames)
    return varnames
