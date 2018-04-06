import io
import re
import json
import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from requests_html import HTMLSession
from zipfile import ZipFile
from multiprocessing import cpu_count

from .utils import mywrap


class npi(object):
    """A class to work with NPI codes"""

    def __init__(
            self,
            columns=None,
            regex=None,
            download: bool = False,
            path: str = '',
            load: bool = True):
        self.num_cpu = cpu_count()

        if download:
            if path == '':
                raise ValueError('If download is True, path must be given')

            path = str(Path(path).expanduser().resolve())

            # Write path location to ~/.medicare_utils.json
            try:
                with open(Path.home() / '.medicare_utils.json') as f:
                    conf = json.load(f)

                conf['npi'] = conf.get('npi', {})
                conf['npi']['data_path'] = path

            except FileNotFoundError:
                conf = {'npi': {'data_path': path}}

            with open(Path.home() / '.medicare_utils.json', 'w') as f:
                json.dump(conf, f)

            Path(conf['npi']['data_path']).mkdir(parents=True, exist_ok=True)
            self.conf = conf
            self._download()

        else:
            try:
                with open(Path.home() / '.medicare_utils.json') as f:
                    self.conf = json.load(f)
            except FileNotFoundError:
                msg = f"""\
                Must download data on first use.
                Use download=True and give path to save data.
                """
                raise FileNotFoundError(mywrap(msg))

        if load:
            self.codes = self.load(columns=columns, regex=regex)

    def _download(self):
        # Get link of latest NPPES NPI file
        session = HTMLSession()
        page = session.get('http://download.cms.gov/nppes/NPI_Files.html')
        a = page.html.find('a')
        regex1 = re.compile(r'data dissemination', re.IGNORECASE).search
        regex2 = re.compile(r'update', re.IGNORECASE).search
        a = [x for x in a if regex1(x.text) and not regex2(x.text)]
        assert len(a) == 1
        href = list(a[0].absolute_links)[0]

        print('Downloading latest NPI file.')
        # r: requests object
        # b: buffer in memory to hold requests object
        #   this is necessary because ZipFile expects file-like object
        # z: ZipFile object
        # f: individual files within ZipFile
        path = Path(self.conf['npi']['data_path'])
        with requests.get(href, stream=True) as r:
            with io.BytesIO() as b:
                total_length = int(r.headers.get('content-length'))
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=(total_length / 1024) + 1):
                    if chunk:
                        b.write(chunk)

                print('\nFinished downloading NPI file.')
                with ZipFile(b, 'r') as z:
                    # Copy Readme and Code Values PDFs to data folder
                    pdfs = [x for x in z.namelist() if x.endswith('.pdf')]
                    for pdf in pdfs:
                        pdf_bytes = z.read(pdf)
                        with open(path / pdf, 'wb') as f:
                            f.write(pdf_bytes)

                    # Find data file
                    csvs = [x for x in z.namelist() if x.endswith('.csv')]
                    regex = re.compile(r'fileheader', re.IGNORECASE).search
                    csv = [x for x in csvs if not regex(x)]
                    assert len(csv) == 1
                    csv = csv[0]

                    with z.open(csv) as f:
                        msg = f"""\
                        Converting NPI csv data to Parquet format. This takes
                        around 20 minutes, but only has to be done once, and
                        then has very fast read speeds.
                        """
                        print(mywrap(msg))
                        df = pd.read_csv(
                            f,
                            dtype=npi_dtypes,
                            engine='c',
                            parse_dates=[
                                'Provider Enumeration Date', 'Last Update Date',
                                'NPI Deactivation Date',
                                'NPI Reactivation Date'],
                            keep_default_na=True)

        def convert_to_snake_case(string):
            string = re.sub(r'\s+\(.+\)\s*$', '', string).lower()
            return re.sub(r'\s+', '_', string)

        df.columns = [convert_to_snake_case(x) for x in df.columns]

        df.to_parquet(path / 'npi.parquet', engine='pyarrow')

        print('Finished Parquet conversion.')

    def load(self, columns=None, regex=None):
        if type(columns) == str:
            columns = [columns]

        path = self.conf['npi']['data_path'] + 'npi.parquet'
        pf = pq.ParquetFile(path)
        pf_cols = pf.schema.names

        if (columns is None) and (regex is None):
            cols = [
                'npi', 'entity_type_code', 'provider_organization_name',
                'provider_business_practice_location_address_city_name',
                'provider_business_practice_location_address_state_name',
                'provider_enumeration_date', 'last_update_date',
                'npi_deactivation_date', 'npi_reactivation_date',
                'provider_gender_code', 'is_sole_proprietor',
                'is_organization_subpart']
        else:
            cols = ['npi']
            if columns is not None:
                invalid_col_names = [x for x in columns if x not in pf_cols]
                if invalid_col_names != []:
                    msg = f'columns provided are invalid: {invalid_col_names}'
                    raise ValueError(msg)

                cols.extend(columns)

            if regex is not None:
                cols.extend([x for x in pf_cols if re.search(regex, x)])

        nthreads = min(self.num_cpu, len(cols))
        df = pf.read(columns=cols, nthreads=nthreads).to_pandas()
        return df.set_index('npi')


class hcpcs(object):
    """A class to work with HCPCS codes"""

    def __init__(self, year: int, path: str = ''):
        self.num_cpu = cpu_count()

        # Check for ~/.medicare_utils.json file
        try:
            with open(Path.home() / '.medicare_utils.json') as f:
                conf = json.load(f)

            if path != '':
                path = str(Path(path).expanduser().resolve())
                conf['hcpcs'] = conf.get('hcpcs', {})
                conf['hcpcs']['data_path'] = path

                with open(Path.home() / '.medicare_utils.json', 'w') as f:
                    json.dump(conf, f)
        except FileNotFoundError:
            if path == '':
                msg = 'path to store data must be given on first use'
                raise FileNotFoundError(msg)

            conf = {'hcpcs': {'data_path': path}}

            with open(Path.home() / '.medicare_utils.json', 'w') as f:
                json.dump(conf, f)

        self.conf = conf

        Path(conf['hcpcs']['data_path']).mkdir(parents=True, exist_ok=True)
        hcpcs_path = Path(conf['hcpcs']['data_path']) / 'hcpcs.parquet'
        try:
            pq.ParquetFile(hcpcs_path)
        except:
            self.download(hcpcs_path=hcpcs_path)

        df = pd.read_parquet(hcpcs_path, engine='pyarrow')
        self.codes = df.loc[df['year'] == year]

    def _download(self, hcpcs_path):
        all_hcpcs = []
        for year in range(2003, 2019):
            all_hcpcs.append(self._download(year))

        df = pd.concat(all_hcpcs, axis=0)
        df.to_parquet(hcpcs_path, engine='pyarrow')

    def _download_single_year(self, year: int):
        """Download HCPCS codes for a given year

        Args:
            year: Year of codes to download
        Returns:
            DataFrame with columns: 'hcpcs', 'desc', 'year'
        """

        url = 'https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/'

        if year == 2003:
            url += 'physicianfeesched/downloads/rvu03_a.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('pprrvu03.csv')
            df = pd.read_csv(
                rvu,
                header=7,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2004:
            url += 'physicianfeesched/downloads/rvu04_a.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU04.csv')
            df = pd.read_csv(
                rvu,
                header=7,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2005:
            url += 'physicianfeesched/downloads/rvu05_a.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU05.csv')
            df = pd.read_csv(
                rvu,
                header=7,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2006:
            url += 'physicianfeesched/downloads/rvu06a.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU06.csv')
            df = pd.read_csv(
                rvu,
                header=7,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2007:
            url += 'PhysicianFeeSched/Downloads/RVU07B.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU07.csv')
            df = pd.read_csv(
                rvu,
                header=8,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2008:
            url += 'PhysicianFeeSched/Downloads/RVU08AB.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU08.csv')
            df = pd.read_csv(
                rvu,
                header=8,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2009:
            url += 'physicianfeesched/downloads/RVU09A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU09.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2010:
            url += 'physicianfeesched/downloads/RVU10AR1.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU10.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2011:
            url += 'PhysicianFeeSched/Downloads/RVU11A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU11_110210(3).csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2012:
            url += 'PhysicianFeeSched/Downloads/RVU12A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU12.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2013:
            url += 'PhysicianFeeSched/Downloads/RVU13A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU13.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')
        elif year == 2014:
            url += 'PhysicianFeeSched/Downloads/RVU14A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU14_V1219.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2015:
            url += 'PhysicianFeeSched/Downloads/RVU15A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU15_V1223c.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2016:
            url += 'PhysicianFeeSched/Downloads/RVU16A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU16_V0122.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2017:
            url += 'PhysicianFeeSched/Downloads/RVU17A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU17_V1219.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        elif year == 2018:
            url += 'PhysicianFeeSched/Downloads/RVU18A.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            rvu = z.open('PPRRVU18_JAN.csv')
            df = pd.read_csv(
                rvu,
                header=9,
                usecols=['HCPCS', 'DESCRIPTION'],
                encoding='Windows-1252')

        df = df.dropna(axis=0, how='any')
        rename_dict = {'HCPCS': 'hcpcs_cd', 'DESCRIPTION': 'desc'}
        df = df.rename(index=str, columns=rename_dict)
        df['year'] = year
        return df


class icd9(object):
    """A class to work with ICD9 codes"""

    def __init__(self, year: int, long: bool = True, path: str = ''):
        self.num_cpu = cpu_count()

        # Check for ~/.medicare_utils.json file
        try:
            with open(Path.home() / '.medicare_utils.json') as f:
                conf = json.load(f)

            if path != '':
                path = str(Path(path).expanduser().resolve())
                conf['icd9'] = conf.get('icd9', {})
                conf['icd9']['data_path'] = path

                with open(Path.home() / '.medicare_utils.json', 'w') as f:
                    json.dump(conf, f)
        except FileNotFoundError:
            if path == '':
                msg = 'path to store data must be given on first use'
                raise FileNotFoundError(msg)

            conf = {'icd9': {'data_path': path}}

            with open(Path.home() / '.medicare_utils.json', 'w') as f:
                json.dump(conf, f)

        self.conf = conf

        Path(conf['icd9']['data_path']).mkdir(parents=True, exist_ok=True)
        icd9_sg_path = Path(conf['icd9']['data_path']) / 'icd9_sg.parquet'
        icd9_dx_path = Path(conf['icd9']['data_path']) / 'icd9_dx.parquet'
        try:
            pq.ParquetFile(icd9_sg_path)
            pq.ParquetFile(icd9_dx_path)
        except:
            self._download(icd9_sg_path=icd9_sg_path, icd9_dx_path=icd9_dx_path)

        sg_cols = ['icd_prcdr_cd', 'year']
        dx_cols = ['icd_dgns_cd', 'year']
        if long:
            sg_cols.append('desc_long')
            dx_cols.append('desc_long')
        else:
            sg_cols.append('desc_short')
            dx_cols.append('desc_short')

        sg = pd.read_parquet(icd9_sg_path, engine='pyarrow', columns=sg_cols)
        dx = pd.read_parquet(icd9_dx_path, engine='pyarrow', columns=dx_cols)

        sg = sg[sg['year'] == year]
        dx = dx[dx['year'] == year]

        sg = sg.set_index('icd_prcdr_cd')
        dx = dx.set_index('icd_dgns_cd')
        self.sg = sg
        self.dx = dx

    def _download(self, icd9_sg_path, icd9_dx_path):
        all_icd9_sg_short = []
        all_icd9_dx_short = []
        for yr in range(2006, 2016):
            dx, sg = self._download_single_year(year=yr, long=False)
            all_icd9_sg_short.append(sg)
            all_icd9_dx_short.append(dx)

        df_sg_short = pd.concat(all_icd9_sg_short, axis=0)
        df_dx_short = pd.concat(all_icd9_dx_short, axis=0)
        df_sg_short = df_sg_short.rename(
            index=str, columns={'desc': 'desc_short'})
        df_dx_short = df_dx_short.rename(
            index=str, columns={'desc': 'desc_short'})

        all_icd9_sg_long = []
        all_icd9_dx_long = []
        for yr in range(2006, 2016):
            dx, sg = self._download_single_year(year=yr, long=True)
            all_icd9_sg_long.append(sg)
            all_icd9_dx_long.append(dx)

        df_sg_long = pd.concat(all_icd9_sg_long, axis=0)
        df_dx_long = pd.concat(all_icd9_dx_long, axis=0)
        df_sg_long = df_sg_long.rename(index=str, columns={'desc': 'desc_long'})
        df_dx_long = df_dx_long.rename(index=str, columns={'desc': 'desc_long'})

        sg = df_sg_short.merge(
            df_sg_long,
            how='inner',
            on=['icd_prcdr_cd', 'year'],
            validate='1:1')
        dx = df_dx_short.merge(
            df_dx_long, how='inner', on=['icd_dgns_cd', 'year'], validate='1:1')

        assert len(sg) == len(df_sg_short) == len(df_sg_long)
        assert len(dx) == len(df_dx_short) == len(df_dx_long)

        sg.to_parquet(icd9_sg_path, engine='pyarrow')
        dx.to_parquet(icd9_dx_path, engine='pyarrow')

    def _download_single_year(self, year: int, long: bool):

        url = 'https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes'
        url += '/Downloads/'

        if year == 2006:
            url += 'v23_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            sg = z.read('I9SG_DESC.txt').decode('latin-1')
            dx = z.read('I9DX_DESC.txt').decode('latin-1')
            sg = pd.read_fwf(
                io.StringIO(sg),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            dx = pd.read_fwf(
                io.StringIO(dx),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2007:
            url += 'v24_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            sg = z.read('I9surgery.txt').decode('latin-1')
            dx = z.read('I9diagnosis.txt').decode('latin-1')
            sg = pd.read_fwf(
                io.StringIO(sg),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            dx = pd.read_fwf(
                io.StringIO(dx),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2008:
            url += 'v25_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            sg = z.read('I9proceduresV25.txt').decode('latin-1')
            dx = z.read('I9diagnosesV25.txt').decode('latin-1')
            sg = pd.read_fwf(
                io.StringIO(sg),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            dx = pd.read_fwf(
                io.StringIO(dx),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2009:
            url += 'v26_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            sg = z.read('V26  I-9 Procedures.txt').decode('latin-1')
            dx = z.read('V26 I-9 Diagnosis.txt').decode('latin-1')
            sg = pd.read_fwf(
                io.StringIO(sg),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            dx = pd.read_fwf(
                io.StringIO(dx),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2010:
            url += 'v27_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            sg = z.read('CMS27_DESC_SHORT_SG.txt').decode('latin-1')
            dx = z.read('CMS27_DESC_SHORT_DX.txt').decode('latin-1')
            sg = pd.read_fwf(
                io.StringIO(sg),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            dx = pd.read_fwf(
                io.StringIO(dx),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2011:
            url += 'cmsv28_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                sg = z.read('CMS28_DESC_SHORT_SG.txt').decode('latin-1')
                dx = z.read('CMS28_DESC_SHORT_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                sg = z.read('CMS28_DESC_LONG_SG.txt').decode('latin-1')
                dx = z.read('CMS28_DESC_LONG_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2012:
            url += 'cmsv29_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                sg = z.read('CMS29_DESC_SHORT_SG.txt').decode('latin-1')
                dx = z.read('CMS29_DESC_SHORT_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                sg = z.read('CMS29_DESC_LONG_SG.txt').decode('latin-1')
                dx = z.read('CMS29_DESC_LONG_DX.101111.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2013:
            url += 'cmsv30_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                sg = z.read('CMS30_DESC_SHORT_SG.txt').decode('latin-1')
                dx = z.read('CMS30_DESC_SHORT_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                sg = z.read('CMS30_DESC_LONG_SG.txt').decode('latin-1')
                dx = z.read('CMS30_DESC_LONG_DX 080612.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2014:
            url += 'cmsv31-master-descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                sg = z.read('CMS31_DESC_SHORT_SG.txt').decode('latin-1')
                dx = z.read('CMS31_DESC_SHORT_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                sg = z.read('CMS31_DESC_LONG_SG.txt').decode('latin-1')
                dx = z.read('CMS31_DESC_LONG_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2015:
            url += 'ICD-9-CM-v32-master-descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                sg = z.read('CMS32_DESC_SHORT_SG.txt').decode('latin-1')
                dx = z.read('CMS32_DESC_SHORT_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                sg = z.read('CMS32_DESC_LONG_SG.txt').decode('latin-1')
                dx = z.read('CMS32_DESC_LONG_DX.txt').decode('latin-1')
                sg = pd.read_fwf(
                    io.StringIO(sg),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                dx = pd.read_fwf(
                    io.StringIO(dx),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        sg = sg.dropna(axis=0, how='any')
        dx = dx.dropna(axis=0, how='any')
        dx['year'] = year
        sg['year'] = year
        return dx, sg


# Define NPI dtypes for csv import
npi_dtypes = {
    'NPI':
        np.int64,
    'Entity Type Code':
        'str',
    'Replacement NPI':
        np.float64,
    'Employer Identification Number (EIN)':
        'str',
    'Provider Organization Name (Legal Business Name)':
        'str',
    'Provider Last Name (Legal Name)':
        'str',
    'Provider First Name':
        'str',
    'Provider Middle Name':
        'str',
    'Provider Name Prefix Text':
        'str',
    'Provider Name Suffix Text':
        'str',
    'Provider Credential Text':
        'str',
    'Provider Other Organization Name':
        'str',
    'Provider Other Organization Name Type Code':
        'str',
    'Provider Other Last Name':
        'str',
    'Provider Other First Name':
        'str',
    'Provider Other Middle Name':
        'str',
    'Provider Other Name Prefix Text':
        'str',
    'Provider Other Name Suffix Text':
        'str',
    'Provider Other Credential Text':
        'str',
    'Provider Other Last Name Type Code':
        'str',
    'Provider First Line Business Mailing Address':
        'str',
    'Provider Second Line Business Mailing Address':
        'str',
    'Provider Business Mailing Address City Name':
        'str',
    'Provider Business Mailing Address State Name':
        'str',
    'Provider Business Mailing Address Postal Code':
        'str',
    'Provider Business Mailing Address Country Code (If outside U.S.)':
        'str',
    'Provider Business Mailing Address Telephone Number':
        'str',
    'Provider Business Mailing Address Fax Number':
        'str',
    'Provider First Line Business Practice Location Address':
        'str',
    'Provider Second Line Business Practice Location Address':
        'str',
    'Provider Business Practice Location Address City Name':
        'str',
    'Provider Business Practice Location Address State Name':
        'str',
    'Provider Business Practice Location Address Postal Code':
        'str',
    'Provider Business Practice Location Address Country Code (If outside U.S.)':
        'str',
    'Provider Business Practice Location Address Telephone Number':
        'str',
    'Provider Business Practice Location Address Fax Number':
        'str',
    'NPI Deactivation Reason Code':
        'str',
    'Provider Gender Code':
        'str',
    'Authorized Official Last Name':
        'str',
    'Authorized Official First Name':
        'str',
    'Authorized Official Middle Name':
        'str',
    'Authorized Official Title or Position':
        'str',
    'Authorized Official Telephone Number':
        'str',
    'Healthcare Provider Taxonomy Code_1':
        'str',
    'Provider License Number_1':
        'str',
    'Provider License Number State Code_1':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_1':
        'str',
    'Healthcare Provider Taxonomy Code_2':
        'str',
    'Provider License Number_2':
        'str',
    'Provider License Number State Code_2':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_2':
        'str',
    'Healthcare Provider Taxonomy Code_3':
        'str',
    'Provider License Number_3':
        'str',
    'Provider License Number State Code_3':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_3':
        'str',
    'Healthcare Provider Taxonomy Code_4':
        'str',
    'Provider License Number_4':
        'str',
    'Provider License Number State Code_4':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_4':
        'str',
    'Healthcare Provider Taxonomy Code_5':
        'str',
    'Provider License Number_5':
        'str',
    'Provider License Number State Code_5':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_5':
        'str',
    'Healthcare Provider Taxonomy Code_6':
        'str',
    'Provider License Number_6':
        'str',
    'Provider License Number State Code_6':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_6':
        'str',
    'Healthcare Provider Taxonomy Code_7':
        'str',
    'Provider License Number_7':
        'str',
    'Provider License Number State Code_7':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_7':
        'str',
    'Healthcare Provider Taxonomy Code_8':
        'str',
    'Provider License Number_8':
        'str',
    'Provider License Number State Code_8':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_8':
        'str',
    'Healthcare Provider Taxonomy Code_9':
        'str',
    'Provider License Number_9':
        'str',
    'Provider License Number State Code_9':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_9':
        'str',
    'Healthcare Provider Taxonomy Code_10':
        'str',
    'Provider License Number_10':
        'str',
    'Provider License Number State Code_10':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_10':
        'str',
    'Healthcare Provider Taxonomy Code_11':
        'str',
    'Provider License Number_11':
        'str',
    'Provider License Number State Code_11':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_11':
        'str',
    'Healthcare Provider Taxonomy Code_12':
        'str',
    'Provider License Number_12':
        'str',
    'Provider License Number State Code_12':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_12':
        'str',
    'Healthcare Provider Taxonomy Code_13':
        'str',
    'Provider License Number_13':
        'str',
    'Provider License Number State Code_13':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_13':
        'str',
    'Healthcare Provider Taxonomy Code_14':
        'str',
    'Provider License Number_14':
        'str',
    'Provider License Number State Code_14':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_14':
        'str',
    'Healthcare Provider Taxonomy Code_15':
        'str',
    'Provider License Number_15':
        'str',
    'Provider License Number State Code_15':
        'str',
    'Healthcare Provider Primary Taxonomy Switch_15':
        'str',
    'Other Provider Identifier_1':
        'str',
    'Other Provider Identifier Type Code_1':
        'str',
    'Other Provider Identifier State_1':
        'str',
    'Other Provider Identifier Issuer_1':
        'str',
    'Other Provider Identifier_2':
        'str',
    'Other Provider Identifier Type Code_2':
        'str',
    'Other Provider Identifier State_2':
        'str',
    'Other Provider Identifier Issuer_2':
        'str',
    'Other Provider Identifier_3':
        'str',
    'Other Provider Identifier Type Code_3':
        'str',
    'Other Provider Identifier State_3':
        'str',
    'Other Provider Identifier Issuer_3':
        'str',
    'Other Provider Identifier_4':
        'str',
    'Other Provider Identifier Type Code_4':
        'str',
    'Other Provider Identifier State_4':
        'str',
    'Other Provider Identifier Issuer_4':
        'str',
    'Other Provider Identifier_5':
        'str',
    'Other Provider Identifier Type Code_5':
        'str',
    'Other Provider Identifier State_5':
        'str',
    'Other Provider Identifier Issuer_5':
        'str',
    'Other Provider Identifier_6':
        'str',
    'Other Provider Identifier Type Code_6':
        'str',
    'Other Provider Identifier State_6':
        'str',
    'Other Provider Identifier Issuer_6':
        'str',
    'Other Provider Identifier_7':
        'str',
    'Other Provider Identifier Type Code_7':
        'str',
    'Other Provider Identifier State_7':
        'str',
    'Other Provider Identifier Issuer_7':
        'str',
    'Other Provider Identifier_8':
        'str',
    'Other Provider Identifier Type Code_8':
        'str',
    'Other Provider Identifier State_8':
        'str',
    'Other Provider Identifier Issuer_8':
        'str',
    'Other Provider Identifier_9':
        'str',
    'Other Provider Identifier Type Code_9':
        'str',
    'Other Provider Identifier State_9':
        'str',
    'Other Provider Identifier Issuer_9':
        'str',
    'Other Provider Identifier_10':
        'str',
    'Other Provider Identifier Type Code_10':
        'str',
    'Other Provider Identifier State_10':
        'str',
    'Other Provider Identifier Issuer_10':
        'str',
    'Other Provider Identifier_11':
        'str',
    'Other Provider Identifier Type Code_11':
        'str',
    'Other Provider Identifier State_11':
        'str',
    'Other Provider Identifier Issuer_11':
        'str',
    'Other Provider Identifier_12':
        'str',
    'Other Provider Identifier Type Code_12':
        'str',
    'Other Provider Identifier State_12':
        'str',
    'Other Provider Identifier Issuer_12':
        'str',
    'Other Provider Identifier_13':
        'str',
    'Other Provider Identifier Type Code_13':
        'str',
    'Other Provider Identifier State_13':
        'str',
    'Other Provider Identifier Issuer_13':
        'str',
    'Other Provider Identifier_14':
        'str',
    'Other Provider Identifier Type Code_14':
        'str',
    'Other Provider Identifier State_14':
        'str',
    'Other Provider Identifier Issuer_14':
        'str',
    'Other Provider Identifier_15':
        'str',
    'Other Provider Identifier Type Code_15':
        'str',
    'Other Provider Identifier State_15':
        'str',
    'Other Provider Identifier Issuer_15':
        'str',
    'Other Provider Identifier_16':
        'str',
    'Other Provider Identifier Type Code_16':
        'str',
    'Other Provider Identifier State_16':
        'str',
    'Other Provider Identifier Issuer_16':
        'str',
    'Other Provider Identifier_17':
        'str',
    'Other Provider Identifier Type Code_17':
        'str',
    'Other Provider Identifier State_17':
        'str',
    'Other Provider Identifier Issuer_17':
        'str',
    'Other Provider Identifier_18':
        'str',
    'Other Provider Identifier Type Code_18':
        'str',
    'Other Provider Identifier State_18':
        'str',
    'Other Provider Identifier Issuer_18':
        'str',
    'Other Provider Identifier_19':
        'str',
    'Other Provider Identifier Type Code_19':
        'str',
    'Other Provider Identifier State_19':
        'str',
    'Other Provider Identifier Issuer_19':
        'str',
    'Other Provider Identifier_20':
        'str',
    'Other Provider Identifier Type Code_20':
        'str',
    'Other Provider Identifier State_20':
        'str',
    'Other Provider Identifier Issuer_20':
        'str',
    'Other Provider Identifier_21':
        'str',
    'Other Provider Identifier Type Code_21':
        'str',
    'Other Provider Identifier State_21':
        'str',
    'Other Provider Identifier Issuer_21':
        'str',
    'Other Provider Identifier_22':
        'str',
    'Other Provider Identifier Type Code_22':
        'str',
    'Other Provider Identifier State_22':
        'str',
    'Other Provider Identifier Issuer_22':
        'str',
    'Other Provider Identifier_23':
        'str',
    'Other Provider Identifier Type Code_23':
        'str',
    'Other Provider Identifier State_23':
        'str',
    'Other Provider Identifier Issuer_23':
        'str',
    'Other Provider Identifier_24':
        'str',
    'Other Provider Identifier Type Code_24':
        'str',
    'Other Provider Identifier State_24':
        'str',
    'Other Provider Identifier Issuer_24':
        'str',
    'Other Provider Identifier_25':
        'str',
    'Other Provider Identifier Type Code_25':
        'str',
    'Other Provider Identifier State_25':
        'str',
    'Other Provider Identifier Issuer_25':
        'str',
    'Other Provider Identifier_26':
        'str',
    'Other Provider Identifier Type Code_26':
        'str',
    'Other Provider Identifier State_26':
        'str',
    'Other Provider Identifier Issuer_26':
        'str',
    'Other Provider Identifier_27':
        'str',
    'Other Provider Identifier Type Code_27':
        'str',
    'Other Provider Identifier State_27':
        'str',
    'Other Provider Identifier Issuer_27':
        'str',
    'Other Provider Identifier_28':
        'str',
    'Other Provider Identifier Type Code_28':
        'str',
    'Other Provider Identifier State_28':
        'str',
    'Other Provider Identifier Issuer_28':
        'str',
    'Other Provider Identifier_29':
        'str',
    'Other Provider Identifier Type Code_29':
        'str',
    'Other Provider Identifier State_29':
        'str',
    'Other Provider Identifier Issuer_29':
        'str',
    'Other Provider Identifier_30':
        'str',
    'Other Provider Identifier Type Code_30':
        'str',
    'Other Provider Identifier State_30':
        'str',
    'Other Provider Identifier Issuer_30':
        'str',
    'Other Provider Identifier_31':
        'str',
    'Other Provider Identifier Type Code_31':
        'str',
    'Other Provider Identifier State_31':
        'str',
    'Other Provider Identifier Issuer_31':
        'str',
    'Other Provider Identifier_32':
        'str',
    'Other Provider Identifier Type Code_32':
        'str',
    'Other Provider Identifier State_32':
        'str',
    'Other Provider Identifier Issuer_32':
        'str',
    'Other Provider Identifier_33':
        'str',
    'Other Provider Identifier Type Code_33':
        'str',
    'Other Provider Identifier State_33':
        'str',
    'Other Provider Identifier Issuer_33':
        'str',
    'Other Provider Identifier_34':
        'str',
    'Other Provider Identifier Type Code_34':
        'str',
    'Other Provider Identifier State_34':
        'str',
    'Other Provider Identifier Issuer_34':
        'str',
    'Other Provider Identifier_35':
        'str',
    'Other Provider Identifier Type Code_35':
        'str',
    'Other Provider Identifier State_35':
        'str',
    'Other Provider Identifier Issuer_35':
        'str',
    'Other Provider Identifier_36':
        'str',
    'Other Provider Identifier Type Code_36':
        'str',
    'Other Provider Identifier State_36':
        'str',
    'Other Provider Identifier Issuer_36':
        'str',
    'Other Provider Identifier_37':
        'str',
    'Other Provider Identifier Type Code_37':
        'str',
    'Other Provider Identifier State_37':
        'str',
    'Other Provider Identifier Issuer_37':
        'str',
    'Other Provider Identifier_38':
        'str',
    'Other Provider Identifier Type Code_38':
        'str',
    'Other Provider Identifier State_38':
        'str',
    'Other Provider Identifier Issuer_38':
        'str',
    'Other Provider Identifier_39':
        'str',
    'Other Provider Identifier Type Code_39':
        'str',
    'Other Provider Identifier State_39':
        'str',
    'Other Provider Identifier Issuer_39':
        'str',
    'Other Provider Identifier_40':
        'str',
    'Other Provider Identifier Type Code_40':
        'str',
    'Other Provider Identifier State_40':
        'str',
    'Other Provider Identifier Issuer_40':
        'str',
    'Other Provider Identifier_41':
        'str',
    'Other Provider Identifier Type Code_41':
        'str',
    'Other Provider Identifier State_41':
        'str',
    'Other Provider Identifier Issuer_41':
        'str',
    'Other Provider Identifier_42':
        'str',
    'Other Provider Identifier Type Code_42':
        'str',
    'Other Provider Identifier State_42':
        'str',
    'Other Provider Identifier Issuer_42':
        'str',
    'Other Provider Identifier_43':
        'str',
    'Other Provider Identifier Type Code_43':
        'str',
    'Other Provider Identifier State_43':
        'str',
    'Other Provider Identifier Issuer_43':
        'str',
    'Other Provider Identifier_44':
        'str',
    'Other Provider Identifier Type Code_44':
        'str',
    'Other Provider Identifier State_44':
        'str',
    'Other Provider Identifier Issuer_44':
        'str',
    'Other Provider Identifier_45':
        'str',
    'Other Provider Identifier Type Code_45':
        'str',
    'Other Provider Identifier State_45':
        'str',
    'Other Provider Identifier Issuer_45':
        'str',
    'Other Provider Identifier_46':
        'str',
    'Other Provider Identifier Type Code_46':
        'str',
    'Other Provider Identifier State_46':
        'str',
    'Other Provider Identifier Issuer_46':
        'str',
    'Other Provider Identifier_47':
        'str',
    'Other Provider Identifier Type Code_47':
        'str',
    'Other Provider Identifier State_47':
        'str',
    'Other Provider Identifier Issuer_47':
        'str',
    'Other Provider Identifier_48':
        'str',
    'Other Provider Identifier Type Code_48':
        'str',
    'Other Provider Identifier State_48':
        'str',
    'Other Provider Identifier Issuer_48':
        'str',
    'Other Provider Identifier_49':
        'str',
    'Other Provider Identifier Type Code_49':
        'str',
    'Other Provider Identifier State_49':
        'str',
    'Other Provider Identifier Issuer_49':
        'str',
    'Other Provider Identifier_50':
        'str',
    'Other Provider Identifier Type Code_50':
        'str',
    'Other Provider Identifier State_50':
        'str',
    'Other Provider Identifier Issuer_50':
        'str',
    'Is Sole Proprietor':
        'str',
    'Is Organization Subpart':
        'str',
    'Parent Organization LBN':
        'str',
    'Parent Organization TIN':
        'str',
    'Authorized Official Name Prefix Text':
        'str',
    'Authorized Official Name Suffix Text':
        'str',
    'Authorized Official Credential Text':
        'str',
    'Healthcare Provider Taxonomy Group_1':
        'str',
    'Healthcare Provider Taxonomy Group_2':
        'str',
    'Healthcare Provider Taxonomy Group_3':
        'str',
    'Healthcare Provider Taxonomy Group_4':
        'str',
    'Healthcare Provider Taxonomy Group_5':
        'str',
    'Healthcare Provider Taxonomy Group_6':
        'str',
    'Healthcare Provider Taxonomy Group_7':
        'str',
    'Healthcare Provider Taxonomy Group_8':
        'str',
    'Healthcare Provider Taxonomy Group_9':
        'str',
    'Healthcare Provider Taxonomy Group_10':
        'str',
    'Healthcare Provider Taxonomy Group_11':
        'str',
    'Healthcare Provider Taxonomy Group_12':
        'str',
    'Healthcare Provider Taxonomy Group_13':
        'str',
    'Healthcare Provider Taxonomy Group_14':
        'str',
    'Healthcare Provider Taxonomy Group_15':
        'str'}
