import json
import pandas as pd


def main():
    for data_type in ['med', 'car', 'ip', 'op', 'dme', 'hha', 'hos', 'snf']:
        if data_type == 'med':
            df = pd.read_stata('harmmed.dta')
        elif data_type == 'car':
            car = pd.read_stata('harmcar.dta')
            carc = pd.read_stata('harmcarc.dta')
            carl = pd.read_stata('harmcarl.dta')
            df = pd.concat([car, carc, carl])
        elif data_type == 'ip':
            ip = pd.read_stata('harmip.dta')
            ipc = pd.read_stata('harmipc.dta')
            ipr = pd.read_stata('harmipr.dta')
            df = pd.concat([ip, ipc, ipr])
        elif data_type == 'op':
            op = pd.read_stata('harmop.dta')
            opc = pd.read_stata('harmopc.dta')
            opr = pd.read_stata('harmopr.dta')
            df = pd.concat([op, opc, opr])
        elif data_type == 'dme':
            dmec = pd.read_stata('harmdmec.dta')
            dmel = pd.read_stata('harmdmel.dta')
            df = pd.concat([dmec, dmel])
        elif data_type in ['hha', 'hos', 'snf']:
            df1 = pd.read_stata(f'harm{data_type}c.dta')
            df2 = pd.read_stata(f'harm{data_type}r.dta')
            df = pd.concat([df1, df2])

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
