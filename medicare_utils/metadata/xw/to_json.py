import json
import pandas as pd

def main():
    """Internal code to convert Jean's crosswalks to JSON files
    """
    data_types = ['carc', 'carl', 'dmec', 'dmel', 'hhac', 'hhar', 'hosc',
        'hosr', 'ipc', 'ipr', 'med', 'opc', 'opr', 'snfc', 'snfr']
    for data_type in data_types:
        df = pd.read_stata(f'harm{data_type}.dta')
        df = df.sort_values(['cname', 'year'])
        df = df[df['year'] >= 1999]
        df = df[df['cname'] != '']
        xw = {}

        for i in range(len(df)):
            dfi = df.iloc[i]

            yeari = str(dfi['year'])
            cnamei = dfi['cname']
            namei = dfi['name']
            typei = dfi['type']
            formati = dfi['format']
            labeli = dfi['varlab']

            xw[cnamei] = xw.get(cnamei, {})
            xw[cnamei]['desc'] = labeli
            xw[cnamei][yeari] = xw[cnamei].get(yeari, {})
            xw[cnamei][yeari]['name'] = xw[cnamei][yeari].get('name', namei)
            xw[cnamei][yeari]['type'] = xw[cnamei][yeari].get('type', typei)
            xw[cnamei][yeari]['format'] = xw[cnamei][yeari].get(
                'format', formati)

        with open(f'{data_type}.json', 'w') as f:
            json.dump(xw, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
