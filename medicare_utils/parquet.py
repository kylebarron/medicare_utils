#! /usr/bin/env python3
import os
import re
import math
import inspect
import pkg_resources
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import fastparquet as fp
from pandas.api.types import CategoricalDtype
from time import time
from joblib import Parallel, delayed
pkg_resources.require("pandas>=0.21.0")


def main(
        pcts=['0001', '01', '05', '100'],
        years=range(2001, 2013),
        med_types=['carc', 'opc', 'bsfab', 'med'],
        n_jobs=6):
    """Main program: In parallel convert Stata files to parquet

    Args:
        pcts: string or list of strings of percent samples to convert
        years: int, range, or list of ints of file years to convert
        med_types: string or list of strings of type of data files to convert
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
        n_jobs: number of processes to use
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

    if type(med_types) is str:
        med_types = [med_types]
    elif type(med_types) is list:
        pass
    else:
        raise TypeError('med_types must be string or list of strings')

    data_list = [[x, y, z] for x in pcts for y in years for z in med_types]

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

    Parallel(n_jobs=n_jobs)(delayed(convert_med)(*i) for i in data_list)


def convert_med(
        pct,
        year,
        data_type,
        rg_size=2.5,
        med_dta='/disk/aging/medicare/data',
        med_pq='/homes/nber/barronk/agebulk1/raw'):
    """Top-level function to convert a given percent sample, year, and
    data type of file to parquet format.

    Args:
        pct: percent sample to convert
        year: year of data to convert
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
        rg_size: size in GB of each Parquet row group
        med_dta: canonical path for raw medicare dta files
        med_pq: top of tree to output new parquet files
    Returns:
        nothing. Writes parquet file to disk.
    Raises:
        NameError if data_type is not one of the above
    """

    year = int(year)

    jroth_dir = '/disk/agedisk2/medicare.work/doyle-DUA18266/jroth'
    if data_type == 'carc':
        varnames = pd.read_stata(f'{jroth_dir}/harmcarc.dta')
        if year >= 2002:
            infile = f'{med_dta}/{pct}pct/car/{year}/carc{year}.dta'
        else:
            infile = f'{med_dta}/{pct}pct/car/{year}/car{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/car/carc{year}.parquet'

    elif data_type == 'carl':
        assert year >= 2002
        varnames = pd.read_stata(f'{jroth_dir}/harmcarl.dta')
        infile = f'{med_dta}/{pct}pct/car/{year}/carl{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/car/carl{year}.parquet'

    elif data_type == 'den':
        varnames = pd.read_stata(f'{jroth_dir}/harmden.dta')
        infile = f'{med_dta}/{pct}pct/den/{year}/den{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/den/den{year}.parquet'

    elif data_type == 'ipc':
        varnames = pd.read_stata(f'{jroth_dir}/harmipc.dta')
        if year >= 2002:
            infile = f'{med_dta}/{pct}pct/ip/{year}/ipc{year}.dta'
        else:
            infile = f'{med_dta}/{pct}pct/ip/{year}/ip{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/ip/ipc{year}.parquet'

    elif data_type == 'ipr':
        assert year >= 2002
        varnames = pd.read_stata(f'{jroth_dir}/harmipr.dta')
        infile = f'{med_dta}/{pct}pct/ip/{year}/ipr{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/ip/ipr{year}.parquet'

    elif data_type == 'med':
        varnames = pd.read_stata(f'{jroth_dir}/harmmed.dta')
        infile = f'{med_dta}/{pct}pct/med/{year}/med{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/med/med{year}.parquet'

    elif data_type == 'op':
        varnames = pd.read_stata(f'{jroth_dir}/harmop.dta')
        raise Exception('Haven\'t added support yet for older op files')

    elif data_type == 'opc':
        # For years 2001(?) - present, use harmopc and harmopr; for before then
        # use harmop
        # harmop has some duplicated varnames from 2001 on, most likely due to
        # the mix of claims and revenue center data

        varnames = pd.read_stata(f'{jroth_dir}/harmopc.dta')
        infile = f'{med_dta}/{pct}pct/op/{year}/opc{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/op/opc{year}.parquet'

    elif data_type == 'opr':
        varnames = pd.read_stata(f'{jroth_dir}/harmopr.dta')
        infile = f'{med_dta}/{pct}pct/op/{year}/opr{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/op/opr{year}.parquet'

    elif data_type == 'bsfab':
        varnames = None
        infile = f'{med_dta}/{pct}pct/bsf/{year}/1/bsfab{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/bsf/bsfab{year}.parquet'

    elif data_type == 'bsfcc':
        varnames = None
        infile = f'{med_dta}/{pct}pct/bsf/{year}/1/bsfcc{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/bsf/bsfcc{year}.parquet'

    elif data_type == 'bsfcu':
        varnames = None
        infile = f'{med_dta}/{pct}pct/bsf/{year}/1/bsfcu{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/bsf/bsfcu{year}.parquet'

    elif data_type == 'bsfd':
        varnames = None
        infile = f'{med_dta}/{pct}pct/bsf/{year}/1/bsfd{year}.dta'
        outfile = f'{med_pq}/pq/{pct}pct/bsf/bsfd{year}.parquet'

    else:
        raise NameError('Unknown data type')

    if varnames is not None:
        if year in set(varnames.year):
            varnames = varnames.loc[varnames['year'] == year]
            rename_dict = varnames.set_index('name').to_dict()['cname']

            # Can't have missing values in rename_dict
            for key, val in rename_dict.items():
                if val == '':
                    rename_dict[key] = key

            # Remove items from dict that map to duplicate values
            # Can't save a parquet file where multiple cols have same name
            rev_rename_dict = {}
            for key, value in rename_dict.items():
                rev_rename_dict.setdefault(value, set()).add(key)
            dups = [key for key, val in rev_rename_dict.items() if len(val) > 1]

            [
                rename_dict.pop(k)
                for k, v in rename_dict.copy().items()
                if v in dups]
        else:
            print(f'Year not in variable dictionary: {year}')
            rename_dict = None
    else:
        rename_dict = None

    # Make folder path if it doesn't exist
    folder = re.search(r'^(.+)/[^/]+$', outfile)[1]
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    msg = f'Starting {data_type} conversion:\n'
    msg += f'\tPercent {pct}\n\tYear {year}'
    print(msg)

    convert_file(
        infile=infile,
        outfile=outfile,
        rename_dict=rename_dict,
        rg_size=rg_size)


def convert_file(
        infile,
        outfile,
        rename_dict=None,
        rg_size=2.5,
        parquet_engine='fastparquet',
        compression_type='SNAPPY',
        manual_schema=False):
    """Convert arbitrary file to Parquet format

    Args:
        infile: path of file to read from
        outfile: path of file to export to
        rename_dict: a dictionary with varnames and canonical varnames
        rg_size: Size in GB of the individual row groups
        parquet_engine: str: either 'pyarrow' or 'fastparquet'
        manual_schema: Whether to make my own schema, for use with pyarrow
    Returns:
        Nothing. Writes .parquet file to disk.
    Raises:
    """
    t0 = time()

    infile_stub = re.search(r'/?([^/]+?)\.([a-z0-9]+)$', infile)
    infile_stub, infile_type = infile_stub[1], infile_stub[2]

    # Set row group size. The following makes an even multiple of row groups
    # as close as possible to the given `rg_size`
    file_size = os.stat(infile).st_size / (1024 ** 3)
    n_rg = round(file_size / rg_size)
    if n_rg == 0:
        n_rg += 1

    nrow_total = pd.read_stata(infile, iterator=True).nobs
    nrow_rg = math.ceil(nrow_total / n_rg)
    gb_per_rg = file_size / n_rg

    msg = f'Row groups:\n\t{n_rg}\n\tof size {gb_per_rg:.2f} GB'
    msg += f'\n\tinfile: {infile_stub}'
    msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
    print(msg)

    msg = f'Beginning scanning dtypes of file\n\tinfile: {infile_stub}'
    msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
    print(msg)

    dtypes = scan_file(infile)
    if rename_dict is not None:
        for old_name, new_name in rename_dict.items():
            try:
                dtypes[new_name] = dtypes.pop(old_name)
            except KeyError:
                pass

    msg = f'Finished scanning dtypes of file\n\tinfile: {infile_stub}'
    msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
    print(msg)

    if parquet_engine == 'pyarrow':
        row = pd.read_stata(infile, iterator=True).get_chunk(1)
        if rename_dict is not None:
            row.rename(index=str, columns=rename_dict, inplace=True)
        row = row.astype(dtypes)

        if manual_schema:
            schema = create_parquet_schema(row.dtypes)
        else:
            schema = pa.Table.from_pandas(row, preserve_index=False).schema
        writer = pq.ParquetWriter(outfile, schema)

    itr = pd.read_stata(infile, chunksize=nrow_rg)
    i = 0
    for df in itr:
        i += 1
        msg = f'Read from file:\n\tGroup {i}'
        msg += f'\n\tinfile: {infile_stub}.{infile_type}'
        msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
        print(msg)

        if rename_dict is not None:
            df.rename(index=str, columns=rename_dict, inplace=True)

        df = df.astype(dtypes)

        msg = f'Cleaned file:\n\tGroup {i}'
        msg += f'\n\tinfile: {infile_stub}'
        msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
        print(msg)

        if parquet_engine == 'pyarrow':
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

        msg = f'Wrote to .parquet:\n\tGroup {i}'
        msg += f'\n\tinfile: {infile_stub}'
        msg += f'\n\ttime: {(time() - t0) / 60:.2f} min'
        print(msg)

    if parquet_engine == 'pyarrow':
        writer.close()

    print('Wrote to .parquet:\n\tAll groups')


def convert_dates(df, datecols):
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


def scan_file(infile, chunksize=100000, cat_threshold=0.1):
    """Scan dta file to find minimal dtypes to hold data in

    For each of the chunks of df:
        for string columns: hold all unique values if I want them categorical
        for float columns: do nothing
        for integer columns: search for missings, highest and lowest value
        for date columns: nothing

    Args:
        infile: dta file to scan
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
        'cat_cols': {
            'names': start_cols['str_cols'],
            'cats': {key: set()
                     for key in start_cols['str_cols']}},
        'str_cols': [],
        'float_cols': start_cols['float_cols']}

    tokeep = []
    tokeep.extend(start_cols['int_cols'])
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
        if end_cols['int_cols']['min'][col] >= 0:
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

    for col in end_cols['cat_cols']['names']:
        dtypes_dict[col] = CategoricalDtype(end_cols['cat_cols']['cats'][col])

    return dtypes_dict


def create_parquet_schema(dtypes):
    """Create parquet schema from Pandas dtypes

    Args:
        dtypes: A dict or Series of dtypes
    Returns:
        pyarrow.Schema
    """
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
        elif np.issubdtype(vartype, np.datetime64):
            fields.append(pa.field(varname, pa.timestamp('ns')))
        elif vartype == object:
            fields.append(pa.field(varname, pa.string()))

    assert len(dtypes) == len(fields)
    schema = pa.schema(fields)
    return schema


main()
