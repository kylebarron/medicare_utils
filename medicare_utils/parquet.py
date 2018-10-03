#! /usr/bin/env python3
import re
import math
import json
import inspect
import pkg_resources
import numpy as np
import pandas as pd

from time import time
from joblib import Parallel, delayed
from typing import Any, Dict, List, Union
from pathlib import Path
from pkg_resources import resource_filename
from pandas.api.types import CategoricalDtype

from .utils import fpath, _mywrap
pkg_resources.require("pandas>=0.21.0")


def convert_med(
        pcts: Union[str, List[str]] = ['0001', '01', '05', '100'],
        years: Union[int, List[int]] = range(2001, 2013),
        data_types: Union[str, List[str]] = ['carc', 'opc', 'bsfab', 'med'],
        rg_size: float = 2.5,
        parquet_engine: str = 'pyarrow',
        compression_type: str = 'SNAPPY',
        manual_schema: bool = False,
        n_jobs: int = 6,
        med_dta: str = '/disk/aging/medicare/data',
        med_pq:
        str = '/disk/agebulk3/medicare.work/doyle-dua51929/barronk-dua51929/raw/pq') -> None:
    """Convert Medicare Stata files to parquet

    Args:
        pcts: percent samples to convert
        years: file years to convert
        data_types:
            types of data files to convert

            - ``bsfab`` (`Beneficiary Summary File, Base segment`_)
            - ``bsfcc`` (`Beneficiary Summary File, Chronic Conditions segment`_)
            - ``bsfcu`` (`Beneficiary Summary File, Cost & Use segment`_)
            - ``bsfd``  (`Beneficiary Summary File, National Death Index segment`_)
            - ``carc``  (`Carrier File, Claims segment`_)
            - ``carl``  (`Carrier File, Line segment`_)
            - ``den``   (Denominator File)
            - ``dmec``  (`Durable Medical Equipment File, Claims segment`_)
            - ``dmel``  (`Durable Medical Equipment File, Line segment`_)
            - ``hhac``  (`Home Health Agency File, Claims segment`_)
            - ``hhar``  (`Home Health Agency File, Revenue Center segment`_)
            - ``hosc``  (`Hospice File, Claims segment`_)
            - ``hosr``  (`Hospice File, Revenue Center segment`_)
            - ``ipc``   (`Inpatient File, Claims segment`_)
            - ``ipr``   (`Inpatient File, Revenue Center segment`_)
            - ``med``   (`MedPAR File`_)
            - ``opc``   (`Outpatient File, Claims segment`_)
            - ``opr``   (`Outpatient File, Revenue Center segment`_)
            - ``snfc``  (`Skilled Nursing Facility File, Claims segment`_)
            - ``snfr``  (`Skilled Nursing Facility File, Revenue Center segment`_)
            - ``xw``    (Crosswalks files for ``ehic`` - ``bene_id``)

            .. _`Beneficiary Summary File, Base segment`: https://kylebarron.github.io/medicare-documentation/resdac/mbsf/#base-abcd-segment_2
            .. _`Beneficiary Summary File, Chronic Conditions segment`: https://kylebarron.github.io/medicare-documentation/resdac/mbsf/#chronic-conditions-segment_2
            .. _`Beneficiary Summary File, Cost & Use segment`: https://kylebarron.github.io/medicare-documentation/resdac/mbsf/#cost-and-use-segment_1
            .. _`Beneficiary Summary File, National Death Index segment`: https://kylebarron.github.io/medicare-documentation/resdac/mbsf/#national-death-index-segment_1
            .. _`Carrier File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/carrier-rif/#carrier-rif_1
            .. _`Carrier File, Line segment`: https://kylebarron.github.io/medicare-documentation/resdac/carrier-rif/#line-file
            .. _`Durable Medical Equipment File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/dme-rif/#durable-medical-equipment-rif_1
            .. _`Durable Medical Equipment File, Line segment`: https://kylebarron.github.io/medicare-documentation/resdac/dme-rif/#line-file
            .. _`Home Health Agency File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/hha-rif/#home-health-agency-rif_1
            .. _`Home Health Agency File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/hha-rif/#revenue-center-file
            .. _`Hospice File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/hospice-rif/#hospice-rif_1
            .. _`Hospice File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/hospice-rif/#revenue-center-file
            .. _`Inpatient File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/ip-rif/#inpatient-rif_1
            .. _`Inpatient File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/ip-rif/#revenue-center-file
            .. _`MedPAR File`: https://kylebarron.github.io/medicare-documentation/resdac/medpar-rif/#medpar-rif_1
            .. _`Outpatient File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/op-rif/#outpatient-rif_1
            .. _`Outpatient File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/op-rif/#revenue-center-file
            .. _`Skilled Nursing Facility File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/snf-rif/#skilled-nursing-facility-rif_1
            .. _`Skilled Nursing Facility File, Revenue Center segment`: https://kylebarron.github.io/medicare-documentation/resdac/snf-rif/#revenue-center-file

        rg_size: size in GB of each Parquet row group
        parquet_engine: either 'fastparquet' or 'pyarrow'
        compression_type: 'SNAPPY' or 'GZIP'
        manual_schema: whether to create manual parquet schema. Doesn't
            always work.
        n_jobs: number of processes to use
        med_dta: top of tree for medicare stata files
        med_pq: top of tree to output new parquet files
    """

    if type(pcts) is str:
        pcts = [pcts]
    elif type(pcts) is list:
        pass
    else:
        raise TypeError('pcts must be string or list of strings')

    if type(years) is int:
        years = [years]
    elif type(years) is list:
        pass
    elif type(years) is range:
        pass
    else:
        raise TypeError('years must be int, range, or list of ints')

    if type(data_types) is str:
        data_types = [data_types]
    elif type(data_types) is list:
        pass
    else:
        raise TypeError('data_types must be string or list of strings')

    data_list = [[x, y, z] for x in pcts for y in years for z in data_types]

    # Drop 100% carrier:
    # data_list = [
    # x for x in data_list if not (x[2] == 'carc') & (x[0] == '100')]

    # Or:
    # Replace 100% carrier with 20% carrier:
    data_list = [['20', x[1], x[2]]
                 if ((x[2] == 'carc') & (x[0] == '100')) else x
                 for x in data_list]

    # Make sure list is unique:
    data_list = sorted([list(x) for x in set(tuple(y) for y in data_list)])

    Parallel(n_jobs=n_jobs)(
        delayed(_convert_med)(
            *i,
            rg_size=rg_size,
            parquet_engine=parquet_engine,
            compression_type=compression_type,
            manual_schema=manual_schema,
            med_dta=med_dta,
            med_pq=med_pq) for i in data_list)


def _convert_med(
        pct: str,
        year: int,
        data_type: Union[str, List[str]],
        rg_size: float = 2.5,
        parquet_engine: str = 'pyarrow',
        compression_type: str = 'SNAPPY',
        manual_schema: bool = False,
        med_dta: str = '/disk/aging/medicare/data',
        med_pq:
        str = '/disk/agebulk3/medicare.work/doyle-dua51929/barronk-dua51929/raw/pq') -> None:
    """Convert a single Medicare file to parquet format.

    Args:
        pct: percent sample to convert
        year: year of data to convert
        data_type:
            type of data files to convert

            - ``bsfab`` Beneficiary Summary File, Base segment
            - ``bsfcc`` Beneficiary Summary File, Chronic Conditions segment
            - ``bsfcu`` Beneficiary Summary File, Cost & Use segment
            - ``bsfd``  Beneficiary Summary File, National Death Index segment
            - ``carc``  Carrier File, Claims segment
            - ``carl``  Carrier File, Line segment
            - ``den``   Denominator File
            - ``dmec``  Durable Medical Equipment File, Claims segment
            - ``dmel``  Durable Medical Equipment File, Line segment
            - ``hhac``  Home Health Agency File, Claims segment
            - ``hhar``  Home Health Agency File, Revenue Center segment
            - ``hosc``  Hospice File, Claims segment
            - ``hosr``  Hospice File, Revenue Center segment
            - ``ipc``   Inpatient File, Claims segment
            - ``ipr``   Inpatient File, Revenue Center segment
            - ``med``   MedPAR File
            - ``opc``   Outpatient File, Claims segment
            - ``opr``   Outpatient File, Revenue Center segment
            - ``snfc``  Skilled Nursing Facility File, Claims segment
            - ``snfr``  Skilled Nursing Facility File, Revenue Center segment
            - ``xw``    Crosswalks files for ``ehic`` - ``bene_id``
        rg_size: size in GB of each Parquet row group
        parquet_engine: either 'fastparquet' or 'pyarrow'
        compression_type: 'SNAPPY' or 'GZIP'
        manual_schema: whether to create manual parquet schema. Doesn't
            always work.
        med_dta: canonical path for raw medicare dta files
        med_pq: top of tree to output new parquet files
    Returns:
        nothing. Writes parquet file to disk.
    Raises:
        NameError if data_type is not one of the above
    """

    if type(pct) != str:
        raise TypeError('pct must be str')
    if type(year) != int:
        raise TypeError('year must be int')

    infile = fpath(percent=pct, year=year, data_type=data_type, dta=True)
    outfile = fpath(
        percent=pct, year=year, data_type=data_type, dta=False, pq_path=med_pq)

    if not data_type.startswith('bsf'):
        path = resource_filename(
            'medicare_utils', f'metadata/xw/{data_type}.json')
        with open(path) as f:
            varnames = json.load(f)

        rename_dict = {}
        for varname, names in varnames.items():
            n = {k: v for k, v in names.items() if k == str(year)}
            if n:
                rename_dict[n[str(year)]['name']] = varname

        if rename_dict:
            # Remove items from dict that map to duplicate values
            # Can't save a parquet file where multiple cols have same name
            rev_rename_dict = {}
            for key, value in rename_dict.items():
                rev_rename_dict.setdefault(value, set()).add(key)
            dups = [key for key, val in rev_rename_dict.items() if len(val) > 1]

            for k, v in rename_dict.copy().items():
                if v in dups:
                    rename_dict.pop(k)
        else:
            print(f'Year not in variable dictionary: {year}')
            rename_dict = None
    else:
        rename_dict = None

    # Make folder path if it doesn't exist
    folder = Path(outfile).parents[0]
    folder.mkdir(exist_ok=True, parents=True)

    msg = f"""\
    Starting {data_type} conversion
    - Percent: {pct}
    - Year {year}
    """
    print(_mywrap(msg))

    convert_file(
        infile=infile,
        outfile=outfile,
        rename_dict=rename_dict,
        rg_size=rg_size,
        parquet_engine=parquet_engine,
        compression_type=compression_type,
        manual_schema=manual_schema)


def convert_file(
        infile: str,
        outfile: str,
        rename_dict: Dict[str, str] = None,
        rg_size: float = 2.5,
        parquet_engine: str = 'pyarrow',
        compression_type: str = 'SNAPPY',
        manual_schema: bool = False) -> None:
    """Convert arbitrary Stata file to Parquet format

    Args:
        infile: path of file to read from
        outfile: path of file to export to
        rename_dict: keys should be initial variable names; values should
            be new variable names
        rg_size: Size in GB of the individual row groups
        parquet_engine: either ``pyarrow`` or ``fastparquet``
        compression_type: Compression algorithm to use. Can be ``SNAPPY`` or
            ``GZIP``.
        manual_schema: Create parquet schema manually. For use with
            pyarrow; doesn't always work
    Returns:
        Writes .parquet file to disk.
    """
    if parquet_engine == 'pyarrow':
        import pyarrow as pa
        import pyarrow.parquet as pq
    elif parquet_engine == 'fastparquet':
        import fastparquet as fp

    t0 = time()

    infile_stub = re.search(r'/?([^/]+?)\.([a-z0-9]+)$', infile)
    infile_stub, infile_type = infile_stub[1], infile_stub[2]

    # Set row group size. The following makes an even multiple of row groups
    # as close as possible to the given `rg_size`
    file_size = Path(infile).stat().st_size / (1024 ** 3)
    n_rg = round(file_size / rg_size)
    if n_rg == 0:
        n_rg += 1

    nrow_total = pd.read_stata(infile, iterator=True).nobs
    nrow_rg = math.ceil(nrow_total / n_rg)
    gb_per_rg = file_size / n_rg

    msg = f"""\
    Row groups:
    - {n_rg} of size {gb_per_rg:.2f} GB
    - infile: {infile_stub}.{infile_type}
    - time: {(time() - t0) / 60:.2f} minutes
    """
    print(_mywrap(msg))

    msg = f"""\
    Beginning scanning dtypes of file:
    - infile: {infile_stub}.{infile_type}
    - time: {(time() - t0) / 60:.2f} minutes
    """
    print(_mywrap(msg))

    if parquet_engine == 'pyarrow':
        dtypes = _scan_file(infile, categorical=False)
    elif parquet_engine == 'fastparquet':
        dtypes = _scan_file(infile, categorical=True)

    if rename_dict is not None:
        for old_name, new_name in rename_dict.items():
            try:
                dtypes[new_name] = dtypes.pop(old_name)
            except KeyError:
                pass

    msg = f"""\
    Finished scanning dtypes of file
    - infile: {infile_stub}.{infile_type}
    - time: {(time() - t0) / 60:.2f} minutes
    """
    print(_mywrap(msg))

    itr = pd.read_stata(infile, chunksize=nrow_rg)
    i = 0
    for df in itr:
        i += 1
        msg = f"""\
        Read from file:
        - Group {i}
        - infile: {infile_stub}.{infile_type}
        - time: {(time() - t0) / 60:.2f} minutes
        """
        print(_mywrap(msg))

        if rename_dict is not None:
            df = df.rename(columns=rename_dict)

        df = df.astype(dtypes)

        msg = f"""\
        Cleaned file:
        - Group {i}
        - infile: {infile_stub}.{infile_type}
        - time: {(time() - t0) / 60:.2f} minutes
        """
        print(_mywrap(msg))

        if parquet_engine == 'pyarrow':
            if i == 1:
                if manual_schema:
                    schema = _create_parquet_schema(df.dtypes)
                else:
                    schema = pa.Table.from_pandas(
                        df, preserve_index=False).schema
                writer = pq.ParquetWriter(outfile, schema, flavor='spark')

            writer.write_table(pa.Table.from_pandas(df, preserve_index=False))
        elif parquet_engine == 'fastparquet':
            if i == 1:
                fp.write(
                    outfile,
                    df,
                    compression=compression_type,
                    has_nulls=False,
                    write_index=False,
                    object_encoding='utf8')
            else:
                fp.write(
                    outfile,
                    df,
                    compression=compression_type,
                    has_nulls=False,
                    write_index=False,
                    object_encoding='utf8',
                    append=True)

        msg = f"""\
        Wrote to parquet:
        - Group {i}
        - infile: {infile_stub}.{infile_type}
        - time: {(time() - t0) / 60:.2f} minutes
        """
        print(_mywrap(msg))

    if parquet_engine == 'pyarrow':
        writer.close()

    print('Wrote to .parquet:\n\tAll groups')


def _convert_dates(df, datecols):
    for col in datecols:
        if not pd.core.dtypes.common.is_datetimelike(df.iloc[:, col]):
            if df[col].dtype == np.number:
                df.iloc[:, col] = pd.to_datetime(
                    df.iloc[:, col],
                    unit='D',
                    origin=pd.Timestamp('1960-01-01'),
                    errors='coerce')
            elif df[col].dtype == 'object':
                df.loc[:, 'from_dt'] = pd.to_datetime(
                    df.loc[:, 'from_dt'], format='%Y-%m-%d', errors='coerce')
    return df


def _scan_file(
        infile: str,
        categorical: bool = True,
        chunksize: int = 100000,
        cat_threshold: float = 0.1,
        unsigned: bool = False) -> Dict[str, Any]:
    """Scan dta file to find minimal dtypes to hold data in

    For each of the chunks of df:
        for string columns: hold all unique values if I want them categorical
        for float columns: do nothing
        for integer columns: search for missings, highest and lowest value
        for date columns: nothing

    Args:
        infile: dta file to scan
        categorical: whether to change strings to categorical
        chunksize: number of rows of infile to read at a time
        cat_threshold: maximum fraction of unique values in order
            to convert to categorical

    Returns:
        dictionary with variable names and dtyplist
    """
    itr = pd.read_stata(infile, iterator=True)
    varlist_df = pd.DataFrame({
        'format': itr.fmtlist,
        'name': itr.varlist,
        'col_size': itr.col_sizes,
        'dtype': itr.dtyplist,
        'label': list(itr.variable_labels().values())})

    start_cols = {}

    date_fmts = ('%tc', '%tC', '%td', '%d', '%tw', '%tm', '%tq', '%th', '%ty')
    date_cols = varlist_df['format'].apply(lambda x: x.startswith(date_fmts))
    date_cols = varlist_df[date_cols]['name'].values.tolist()
    start_cols['date_cols'] = date_cols

    int_cols = varlist_df['dtype'].apply(
        lambda x: np.issubdtype(x, np.integer) if inspect.isclass(x) else False)
    int_cols = varlist_df[int_cols]['name'].values.tolist()
    int_cols = sorted(list(set(int_cols) - set(date_cols)))
    start_cols['int_cols'] = int_cols

    regex = r'%.+s'
    str_cols = varlist_df['format'].apply(lambda x: bool(re.search(regex, x)))
    str_cols = varlist_df[str_cols]['name'].values.tolist()
    start_cols['str_cols'] = str_cols

    float_cols = varlist_df['dtype'].apply(
        lambda x: np.issubdtype(x, np.floating) if inspect.isclass(x) else False
    )
    float_cols = varlist_df[float_cols]['name'].values.tolist()
    start_cols['float_cols'] = float_cols

    end_cols = {
        'date_cols': start_cols['date_cols'],
        'int_cols': {
            'names': start_cols['int_cols'],
            'min': {key: None
                    for key in start_cols['int_cols']},
            'max': {key: None
                    for key in start_cols['int_cols']}},
        'float_cols': start_cols['float_cols']}
    if categorical:
        end_cols['cat_cols'] = {
            'names': start_cols['str_cols'],
            'cats': {key: set()
                     for key in start_cols['str_cols']}}
        end_cols['str_cols'] = []
    else:
        end_cols['cat_cols'] = {}
        end_cols['str_cols'] = start_cols['str_cols']

    tokeep = []
    tokeep.extend(start_cols['int_cols'])
    if categorical:
        tokeep.extend(start_cols['str_cols'])
    itr = pd.read_stata(infile, columns=tokeep, chunksize=chunksize)

    i = 0
    for df in itr:
        i += 1
        print(f'Scanning group {i} of data')
        # Integer vars:
        int_cols = end_cols['int_cols']['names'].copy()
        for col in int_cols:
            # Check missings
            if df.loc[:, col].isnull().values.any():
                # If missings, convert to float
                end_cols['float_cols'].append(col)
                end_cols['int_cols']['names'].remove(col)
                end_cols['int_cols']['max'].pop(col)
                end_cols['int_cols']['min'].pop(col)
            else:
                # Check minimum
                minval = min(df.loc[:, col])
                if end_cols['int_cols']['min'][col] is None:
                    end_cols['int_cols']['min'][col] = minval
                elif minval < end_cols['int_cols']['min'][col]:
                    end_cols['int_cols']['min'][col] = minval

                # Check maximum
                maxval = max(df.loc[:, col])
                if end_cols['int_cols']['max'][col] is None:
                    end_cols['int_cols']['max'][col] = maxval
                elif maxval > end_cols['int_cols']['max'][col]:
                    end_cols['int_cols']['max'][col] = maxval

        if categorical:
            # Scan str vars for categories
            cat_cols = end_cols['cat_cols']['names'].copy()
            for col in cat_cols:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])

                if num_unique_values / num_total_values < cat_threshold:
                    # Then stays as category
                    # Add category values
                    unique_vals = df[col].unique().tolist()
                    end_cols['cat_cols']['cats'][col].update(unique_vals)
                else:
                    print(f'{col} is now a string')
                    # Becomes regular string column
                    end_cols['str_cols'].append(col)
                    end_cols['cat_cols']['cats'].pop(col)
                    end_cols['cat_cols']['names'].remove(col)

        # Not currently scanning date or float vars

    dtypes_dict = {}

    # Int dtypes:
    for col in end_cols['int_cols']['names']:
        if unsigned and (end_cols['int_cols']['min'][col] >= 0):
            if end_cols['int_cols']['max'][col] <= np.iinfo(np.uint8).max:
                dtypes_dict[col] = np.uint8
            elif end_cols['int_cols']['max'][col] <= np.iinfo(np.uint16).max:
                dtypes_dict[col] = np.uint16
            elif end_cols['int_cols']['max'][col] <= np.iinfo(np.uint32).max:
                dtypes_dict[col] = np.uint32
            elif end_cols['int_cols']['max'][col] <= np.iinfo(np.uint64).max:
                dtypes_dict[col] = np.uint64
        else:
            if False:
                pass
            elif ((end_cols['int_cols']['max'][col] <= np.iinfo(np.int8).max) &
                  (end_cols['int_cols']['min'][col] >= np.iinfo(np.int8).min)):
                dtypes_dict[col] = np.int8
            elif ((end_cols['int_cols']['max'][col] <= np.iinfo(np.int16).max) &
                  (end_cols['int_cols']['min'][col] >= np.iinfo(np.int16).min)):
                dtypes_dict[col] = np.int16
            elif ((end_cols['int_cols']['max'][col] <= np.iinfo(np.int32).max) &
                  (end_cols['int_cols']['min'][col] >= np.iinfo(np.int32).min)):
                dtypes_dict[col] = np.int32
            elif ((end_cols['int_cols']['max'][col] <= np.iinfo(np.int64).max) &
                  (end_cols['int_cols']['min'][col] >= np.iinfo(np.int64).min)):
                dtypes_dict[col] = np.int64

    for col in end_cols['float_cols']:
        dtypes_dict[col] = np.float64

    if categorical:
        for col in end_cols['cat_cols']['names']:
            dtypes_dict[col] = CategoricalDtype(
                end_cols['cat_cols']['cats'][col])

    return dtypes_dict


def _create_parquet_schema(dtypes):
    """Create parquet schema from Pandas dtypes

    Args:
        dtypes: A dict or Series of dtypes
    Returns:
        pyarrow.Schema
    """
    import pyarrow as pa

    dtypes = dict(dtypes)
    fields = []
    for varname, vartype in dtypes.items():
        if vartype == np.float16:
            fields.append(pa.field(varname, pa.float16()))
        elif vartype == np.float32:
            fields.append(pa.field(varname, pa.float32()))
        elif vartype == np.float64:
            fields.append(pa.field(varname, pa.float64()))
        elif vartype == np.int8:
            fields.append(pa.field(varname, pa.int8()))
        elif vartype == np.int16:
            fields.append(pa.field(varname, pa.int16()))
        elif vartype == np.int32:
            fields.append(pa.field(varname, pa.int32()))
        elif vartype == np.int64:
            fields.append(pa.field(varname, pa.int64()))
        elif vartype == np.uint8:
            fields.append(pa.field(varname, pa.uint8()))
        elif vartype == np.uint16:
            fields.append(pa.field(varname, pa.uint16()))
        elif vartype == np.uint32:
            fields.append(pa.field(varname, pa.uint32()))
        elif vartype == np.uint64:
            fields.append(pa.field(varname, pa.uint64()))
        elif vartype == np.bool_:
            fields.append(pa.field(varname, pa.bool_()))
        elif (vartype == object) | (vartype.name == 'category'):
            fields.append(pa.field(varname, pa.string()))
        elif np.issubdtype(vartype, np.datetime64):
            fields.append(pa.field(varname, pa.timestamp('ns')))

    assert len(dtypes) == len(fields)
    schema = pa.schema(fields)
    return schema


if __name__ == '__main__':
    convert_med()
