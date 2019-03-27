#! /usr/bin/env python3
from pathlib import Path
from textwrap import dedent, fill
from typing import Union

allowed_pcts = ['0001', '01', '05', '20', '100']
pct_dict = {0.01: '0001', 1: '01', 5: '05', 20: '20', 100: '100'}


def pq_vars(ParquetFile):
    return ParquetFile.schema.names


def _mywrap(text: str) -> str:
    text = dedent(text)
    lines = text.split('\n')
    lines = [
        fill(x, replace_whitespace=False, subsequent_indent='    ')
        for x in lines]
    text = '\n'.join(lines)
    return text


def fpath(percent, year, data_type, root_path, extension, new_style):
    """Generate path to Medicare files

    Args:
        percent:
            percent sample of data. Can be {'0001', '01', '05', '20', '100'}
        year: year of data.
        data_type:
            desired type of file

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
            - ``xw_bsf`` (Crosswalks files for ``ehic`` - ``bene_id``)

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

        root_path: top of tree for file path
        extension: file extension
        new_style:
            If False, matches the file names at /disk/aging/medicare/data, if
            True, uses simplified directory structure.
    Returns:
        (str) path to file
    """

    # Check types
    if type(data_type) != str:
        raise TypeError('data_type must be str')

    try:
        year = int(year)
    except ValueError:
        raise TypeError('Invalid year provided')

    allowed_pcts = ['0001', '01', '05', '20', '100']
    if percent not in allowed_pcts:
        msg = f'percent must be one of: {allowed_pcts}'
        raise ValueError(msg)

    if extension == '':
        raise ValueError('Must provide valid extension')

    if extension[0] != '.':
        extension = '.' + extension

    root_path = Path(root_path).expanduser().resolve()
    root_path /= f'{percent}pct'
    if data_type in ['bsfab', 'bsfcc', 'bsfcu', 'bsfd', 'carc', 'carl', 'den',
                     'dmec', 'dmel', 'hhac', 'hhar', 'hosc', 'hosr', 'med',
                     'snfc', 'snfr']:
        root_path /= data_type[:3]
    elif data_type == 'xw_bsf' and not new_style:
        root_path /= 'bsf'
    else:
        root_path /= data_type[:2]

    if new_style:
        root_path /= f'{year}'
        if data_type == 'xw':
            root_path /= f'ehicbenex_one{year}{extension}'
        elif data_type == 'xw_bsf':
            root_path /= f'ehicbenex_unique{year}{extension}'
        else:
            root_path /= f'{data_type}{year}{extension}'
    else:
        root_path /= f'{year}'
        if data_type in ['den', 'dmec', 'dmel', 'hhac', 'hhar', 'hosc', 'hosr',
                         'med', 'snfc', 'snfr']:
            root_path /= f'{data_type}{year}{extension}'

        elif data_type in ['bsfab', 'bsfcc', 'bsfcu', 'bsfd']:
            root_path /= f'1/{data_type}{year}{extension}'

        elif data_type in ['carc', 'carl', 'ipc', 'ipr']:
            if year >= 2002:
                root_path /= f'{data_type}{year}{extension}'
            else:
                root_path /= f'{data_type[:-1]}{year}{extension}'

        elif data_type in ['opc', 'opr']:
            if year >= 2001:
                root_path /= f'{data_type}{year}{extension}'
            else:
                root_path /= f'{data_type[:-1]}{year}{extension}'

        elif data_type == 'xw':
            root_path /= f'ehicbenex_one{year}{extension}'

        elif data_type == 'xw_bsf':
            root_path /= f'xw/ehicbenex_unique{year}{extension}'

        else:
            raise ValueError(f'Invalid data_type: {data_type}')

    return str(root_path)
