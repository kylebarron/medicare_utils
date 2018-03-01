#! /usr/bin/env python3

"""Main module."""


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

    import re
    import numpy as np
    import pandas as pd
    import fastparquet as fp

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


def search_for_codes(pct, year, data_type, bene_ids_to_filter=None,
                     hcpcs_codes=None, icd9_diag_codes=None,
                     icd9_proc_codes=None, collapse_codes=False):
    """Search in given dataset for HCPCS/ICD9 codes

    Note: Each code given must be distinct, or collapse_codes must be True

    Args:
        pct (str): percent sample of data to use
        years (int): year of data to search
        data_type (str): One of carc, carl, ipc, ipr, med, opc, opr
        bene_ids_to_filter (Index, list): List of bene_ids to search over
        hcpcs_codes (list[str]): List of HCPCS codes to look for
        icd9_diag_codes (list[str]): List of ICD-9 diagnosis codes to look for
        icd9_proc_codes (list[str]): List of ICD-9 procedure codes to look for
        collapse_codes (bool): If True, returns a single column "match";
            else it returns a column for each code provided

    Returns:
        DataFrame with bene_id and bool columns for each code to search for
    """
    import re
    import pandas as pd
    import fastparquet as fp

    if data_type not in ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']:
        msg = 'data_type provided that does not match any dataset'
        raise ValueError(msg)

    if hcpcs_codes is not None:
        if data_type in ['carc', 'ipc', 'med', 'opc']:
            msg = 'data_type was supplied that does not have HCPCS columns'
            raise ValueError(msg)
    if icd9_diag_codes is not None:
        if data_type in ['ipr', 'opr']:
            msg = 'data_type was supplied that does not have columns for ICD-9'
            msg += 'diagnosis codes'
            raise ValueError(msg)
    if icd9_proc_codes is not None:
        if data_type in ['carc', 'carl', 'ipr', 'opr']:
            msg = 'data_type was supplied that does not have columns for ICD-9'
            msg += 'procedure codes'
            raise ValueError(msg)

    pf = fp.ParquetFile(fpath(pct, year, data_type))

    regex_string = []
    if hcpcs_codes is not None:
        hcpcs_regex = r'^hcpcs_cd$'
        regex_string.append(hcpcs_regex)

    if icd9_diag_codes is not None:
        if data_type == 'carl':
            icd9_diag_regex = r'icd_dgns_cd\d*$'
        elif data_type == 'med':
            icd9_diag_regex = r'^dgnscd\d+$$'
        else:
            icd9_diag_regex = r'^icd_dgns_cd\d+$'
        regex_string.append(icd9_diag_regex)

    if icd9_proc_codes is not None:
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

        if hcpcs_codes is not None:
            hcpcs_cols = [x for x in df if re.search(hcpcs_regex, x)]

        if icd9_diag_codes is not None:
            icd9_diag_cols = [x for x in df if re.search(icd9_diag_regex, x)]

        if icd9_proc_codes is not None:
            icd9_proc_cols = [x for x in df if re.search(icd9_proc_regex, x)]

        if not collapse_codes:
            if hcpcs_codes is not None:
                for code in hcpcs_codes:
                    demo[code] = False
                    idx = df.index[(df[hcpcs_cols] == code).any(axis=1)]
                    demo.loc[idx, code] = True
                df.drop(hcpcs_cols, axis=1, inplace=True)

            if icd9_diag_codes is not None:
                for code in icd9_diag_codes:
                    demo[code] = False
                    idx = df.index[(df[icd9_diag_cols] == code).any(axis=1)]
                    demo.loc[idx, code] = True
                df.drop(icd9_diag_cols, axis=1, inplace=True)

            if icd9_proc_codes is not None:
                for code in icd9_proc_codes:
                    demo[code] = False
                    idx = df.index[(df[icd9_proc_cols] == code).any(axis=1)]
                    demo.loc[idx, code] = True
                df.drop(icd9_proc_cols, axis=1, inplace=True)

        else:
            demo['match'] = False
            if hcpcs_codes is not None:
                idx = df.index[(df[hcpcs_cols].isin(hcpcs_codes)).any(axis=1)]
                demo.loc[idx, 'match'] = True
                df.drop(hcpcs_cols, axis=1, inplace=True)

            if icd9_diag_codes is not None:
                idx = df.index[(
                    df[icd9_diag_cols].isin(icd9_diag_codes)).any(axis=1)]
                demo.loc[idx, 'match'] = True
                df.drop(icd9_diag_cols, axis=1, inplace=True)

            if icd9_proc_codes is not None:
                idx = df.index[(
                    df[icd9_proc_cols].isin(icd9_proc_codes)).any(axis=1)]
                demo.loc[idx, 'match'] = True
                df.drop(icd9_proc_cols, axis=1, inplace=True)

        all_demo.append(demo)

    if len(all_demo) == 1:
        demo = all_demo[0]
    else:
        demo_concat = pd.concat(all_demo, axis=1, join='outer')

        if not collapse_codes:
            demo = pd.DataFrame(index=demo_concat.index)
            if hcpcs_codes is not None:
                for code in hcpcs_codes:
                    demo[code] = demo_concat[code].max(axis=1)

            if icd9_diag_codes is not None:
                for code in icd9_diag_codes:
                    demo[code] = demo_concat[code].max(axis=1)

            if icd9_proc_codes is not None:
                for code in icd9_proc_codes:
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
