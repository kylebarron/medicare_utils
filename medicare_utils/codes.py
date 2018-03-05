import pandas as pd


class hcpcs(object):
    """A class to work with HCPCS codes"""

    def __init__(self, year):
        self.codes = self._download(year)

    def _download(self, year: int):
        """Download HCPCS codes for a given year

        Args:
            year: Year of codes to download
        Returns:
            DataFrame with columns: 'hcpcs', 'desc', 'year'
        """
        import requests
        import io
        from zipfile import ZipFile

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
        df = df.set_index('hcpcs_cd')
        return df


class icd9(object):
    """A class to work with ICD9 codes"""

    def __init__(self, year: int, long: bool = True):
        self.diag, self.proc = self._download(year, long)

    def _download(self, year: int, long: bool):
        import requests
        import io
        from zipfile import ZipFile

        url = 'https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes'
        url += '/Downloads/'

        if year == 2006:
            url += 'v23_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            proc = z.read('I9SG_DESC.txt').decode('latin-1')
            diag = z.read('I9DX_DESC.txt').decode('latin-1')
            proc = pd.read_fwf(
                io.StringIO(proc),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            diag = pd.read_fwf(
                io.StringIO(diag),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2007:
            url += 'v24_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            proc = z.read('I9surgery.txt').decode('latin-1')
            diag = z.read('I9diagnosis.txt').decode('latin-1')
            proc = pd.read_fwf(
                io.StringIO(proc),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            diag = pd.read_fwf(
                io.StringIO(diag),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2008:
            url += 'v25_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            proc = z.read('I9proceduresV25.txt').decode('latin-1')
            diag = z.read('I9diagnosesV25.txt').decode('latin-1')
            proc = pd.read_fwf(
                io.StringIO(proc),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            diag = pd.read_fwf(
                io.StringIO(diag),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2009:
            url += 'v26_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            proc = z.read('V26  I-9 Procedures.txt').decode('latin-1')
            diag = z.read('V26 I-9 Diagnosis.txt').decode('latin-1')
            proc = pd.read_fwf(
                io.StringIO(proc),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            diag = pd.read_fwf(
                io.StringIO(diag),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2010:
            url += 'v27_icd9.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            proc = z.read('CMS27_DESC_SHORT_SG.txt').decode('latin-1')
            diag = z.read('CMS27_DESC_SHORT_DX.txt').decode('latin-1')
            proc = pd.read_fwf(
                io.StringIO(proc),
                widths=[5, 200],
                names=['icd_prcdr_cd', 'desc'],
                dtype={'icd_prcdr_cd': 'str'})
            diag = pd.read_fwf(
                io.StringIO(diag),
                widths=[5, 200],
                names=['icd_dgns_cd', 'desc'],
                dtype={'icd_dgns_cd': 'str'})

        elif year == 2011:
            url += 'cmsv28_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                proc = z.read('CMS28_DESC_SHORT_SG.txt').decode('latin-1')
                diag = z.read('CMS28_DESC_SHORT_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                proc = z.read('CMS28_DESC_LONG_SG.txt').decode('latin-1')
                diag = z.read('CMS28_DESC_LONG_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2012:
            url += 'cmsv29_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                proc = z.read('CMS29_DESC_SHORT_SG.txt').decode('latin-1')
                diag = z.read('CMS29_DESC_SHORT_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                proc = z.read('CMS29_DESC_LONG_SG.txt').decode('latin-1')
                diag = z.read('CMS29_DESC_LONG_DX.101111.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2013:
            url += 'cmsv30_master_descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                proc = z.read('CMS30_DESC_SHORT_SG.txt').decode('latin-1')
                diag = z.read('CMS30_DESC_SHORT_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                proc = z.read('CMS30_DESC_LONG_SG.txt').decode('latin-1')
                diag = z.read('CMS30_DESC_LONG_DX 080612.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2014:
            url += 'cmsv31-master-descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                proc = z.read('CMS31_DESC_SHORT_SG.txt').decode('latin-1')
                diag = z.read('CMS31_DESC_SHORT_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                proc = z.read('CMS31_DESC_LONG_SG.txt').decode('latin-1')
                diag = z.read('CMS31_DESC_LONG_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        elif year == 2015:
            url += 'ICD-9-CM-v32-master-descriptions.zip'
            content = requests.get(url).content

            z = ZipFile(io.BytesIO(content), 'r')
            if not long:
                proc = z.read('CMS32_DESC_SHORT_SG.txt').decode('latin-1')
                diag = z.read('CMS32_DESC_SHORT_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

            else:
                proc = z.read('CMS32_DESC_LONG_SG.txt').decode('latin-1')
                diag = z.read('CMS32_DESC_LONG_DX.txt').decode('latin-1')
                proc = pd.read_fwf(
                    io.StringIO(proc),
                    widths=[5, 200],
                    names=['icd_prcdr_cd', 'desc'],
                    dtype={'icd_prcdr_cd': 'str'})
                diag = pd.read_fwf(
                    io.StringIO(diag),
                    widths=[5, 200],
                    names=['icd_dgns_cd', 'desc'],
                    dtype={'icd_dgns_cd': 'str'})

        proc = proc.dropna(axis=0, how='any')
        diag = diag.dropna(axis=0, how='any')
        proc = proc.set_index('icd_prcdr_cd')
        diag = diag.set_index('icd_dgns_cd')
        return diag, proc
