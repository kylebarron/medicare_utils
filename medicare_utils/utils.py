#! /usr/bin/env python3
from pathlib import Path
from textwrap import dedent, fill

allowed_pcts = ['0001', '01', '05', '20', '100']
pct_dict = {0.01: '0001', 1: '01', 5: '05', 20: '20', 100: '100'}


def pq_vars(ParquetFile):
    return ParquetFile.schema.names


def _mywrap(text):
    text = dedent(text)
    lines = text.split('\n')
    lines = [fill(x, replace_whitespace=False, subsequent_indent='\t') for x in lines]
    text = '\n'.join(lines)
    return text


def fpath(
        percent,
        year: int,
        data_type: str,
        dta: bool = False,
        dta_path: str = '/disk/aging/medicare/data',
        pq_path: str = '/homes/nber/barronk/agebulk1/raw/pq'):
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
        dta_path: top of tree for medicare stata files
        pq_path: top of tree for medicare parquet files
    Returns:
        string with file path.
    Raises:
        NameError if data_type is not one of the above
    """

    # Check types
    if type(data_type) != str:
        raise TypeError('data_type must be str')

    try:
        year = int(year)
    except ValueError:
        raise TypeError('year must be int')

    if (type(percent) == float) or (type(percent) == int):
        try:
            percent = pct_dict[percent]
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

    allowed_data_types = [
        'bsfab', 'bsfcc', 'bsfcu', 'bsfd', 'carc', 'carl', 'den', 'dmec',
        'dmel', 'hhac', 'hhar', 'hosc', 'hosr', 'ipc', 'ipr', 'med', 'opc',
        'opr', 'snfc', 'snfr', 'xw']
    if data_type not in allowed_data_types:
        raise ValueError(f'data_type must be one of:\n{allowed_data_types}')

    dta_path = str(Path(dta_path).expanduser().resolve())
    pq_path = str(Path(pq_path).expanduser().resolve())

    if data_type == 'bsfab':
        dta_path = f'{dta_path}/{percent}pct/bsf/{year}/1/bsfab{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/bsf/bsfab{year}.parquet'
    elif data_type == 'bsfcc':
        dta_path = f'{dta_path}/{percent}pct/bsf/{year}/1/bsfcc{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/bsf/bsfcc{year}.parquet'
    elif data_type == 'bsfcu':
        dta_path = f'{dta_path}/{percent}pct/bsf/{year}/1/bsfcu{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/bsf/bsfcu{year}.parquet'
    elif data_type == 'bsfd':
        dta_path = f'{dta_path}/{percent}pct/bsf/{year}/1/bsfd{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/bsf/bsfd{year}.parquet'

    elif data_type == 'carc':
        if year >= 2002:
            dta_path = f'{dta_path}/{percent}pct/car/{year}/carc{year}.dta'
        else:
            dta_path = f'{dta_path}/{percent}pct/car/{year}/car{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/car/carc{year}.parquet'
    elif data_type == 'carl':
        assert year >= 2002
        dta_path = f'{dta_path}/{percent}pct/car/{year}/carl{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/car/carl{year}.parquet'

    elif data_type == 'den':
        dta_path = f'{dta_path}/{percent}pct/den/{year}/den{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/den/den{year}.parquet'

    elif data_type == 'dmec':
        dta_path = f'{dta_path}/{percent}pct/dme/{year}/dmec{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/dme/dmec{year}.parquet'
    elif data_type == 'dmel':
        dta_path = f'{dta_path}/{percent}pct/dme/{year}/dmel{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/dme/dmel{year}.parquet'

    elif data_type == 'hhac':
        dta_path = f'{dta_path}/{percent}pct/hha/{year}/hhac{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/hha/hhac{year}.parquet'
    elif data_type == 'hhar':
        dta_path = f'{dta_path}/{percent}pct/hha/{year}/hhar{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/hha/hhar{year}.parquet'

    elif data_type == 'hosc':
        dta_path = f'{dta_path}/{percent}pct/hos/{year}/hosc{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/hos/hosc{year}.parquet'
    elif data_type == 'hosr':
        dta_path = f'{dta_path}/{percent}pct/hos/{year}/hosr{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/hos/hosr{year}.parquet'

    elif data_type == 'ipc':
        if year >= 2002:
            dta_path = f'{dta_path}/{percent}pct/ip/{year}/ipc{year}.dta'
        else:
            dta_path = f'{dta_path}/{percent}pct/ip/{year}/ip{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/ip/ipc{year}.parquet'
    elif data_type == 'ipr':
        assert year >= 2002
        dta_path = f'{dta_path}/{percent}pct/ip/{year}/ipr{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/ip/ipr{year}.parquet'

    elif data_type == 'med':
        dta_path = f'{dta_path}/{percent}pct/med/{year}/med{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/med/med{year}.parquet'

    elif data_type == 'opc':
        dta_path = f'{dta_path}/{percent}pct/op/{year}/opc{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/op/opc{year}.parquet'
    elif data_type == 'opr':
        dta_path = f'{dta_path}/{percent}pct/op/{year}/opr{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/op/opr{year}.parquet'

    elif data_type == 'snfc':
        dta_path = f'{dta_path}/{percent}pct/snf/{year}/snfc{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/snf/snfc{year}.parquet'
    elif data_type == 'snfr':
        dta_path = f'{dta_path}/{percent}pct/snf/{year}/snfr{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/snf/snfr{year}.parquet'

    elif data_type == 'xw':
        dta_path = f'{dta_path}/{percent}pct/xw/{year}/ehicbenex_one{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/xw/ehicbenex_one{year}.parquet'

    if dta:
        return dta_path
    else:
        return pq_path
