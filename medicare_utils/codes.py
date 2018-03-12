import pandas as pd

# class npi(object):
#     """A class to work with NPI codes"""
#


def npi_to_parquet():
    import re
    import numpy as np

    path = '/homes/nber/barronk/agebulk1/raw/codes/npi/'
    path += 'npidata_20050523-20180213.csv'

    dtypes = {
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

    df = pd.read_csv(
        path,
        dtype=dtypes,
        engine='c',
        parse_dates=[
            'Provider Enumeration Date', 'Last Update Date',
            'NPI Deactivation Date', 'NPI Reactivation Date'],
        keep_default_na=True)

    def convert_to_snake_case(string):
        string = re.sub(r'\s+\(.+\)\s*$', '', string).lower()
        return re.sub(r'\s+', '_', string)

    df.columns = [convert_to_snake_case(x) for x in df.columns]

    path = '/disk/agebulk1/medicare.work/doyle-DUA18266/barronk/raw/codes/npi'
    path += '/npidata.parquet'
    df.to_parquet(path)


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
