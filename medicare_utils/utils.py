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


def fpath(
        percent: Union[float, int, str],
        year: Union[int, str],
        data_type: str,
        dta: bool = False,
        dta_path: str = '/disk/aging/medicare/data',
        pq_path:
        str = '/disk/agebulk3/medicare.work/doyle-dua51929/barronk-dua51929/raw/pq'
        ) -> str:
    """Generate path to Medicare files

    Args:
        percent: percent sample of data
        year: year of data
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

        dta: Returns Stata file path
        dta_path: top of tree for medicare stata files
        pq_path: top of tree for medicare parquet files
    Returns:
        path to file
    """

    # Check types
    if type(data_type) != str:
        raise TypeError('data_type must be str')

    try:
        year = int(year)
    except ValueError:
        raise TypeError('Invalid year provided')

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

    elif data_type == 'xw_bsf':
        dta_path = f'{dta_path}/{percent}pct/bsf/{year}/xw/ehic2bene_id{year}.dta'
        pq_path = f'{pq_path}/{percent}pct/xw/ehic2bene_id{year}.parquet'

    else:
        allowed_data_types = [
            'bsfab', 'bsfcc', 'bsfcu', 'bsfd', 'carc', 'carl', 'den', 'dmec',
            'dmel', 'hhac', 'hhar', 'hosc', 'hosr', 'ipc', 'ipr', 'med', 'opc',
            'opr', 'snfc', 'snfr', 'xw', 'xw_bsf']
        raise ValueError(f'data_type must be one of:\n{allowed_data_types}')

    if dta:
        return dta_path
    else:
        return pq_path
