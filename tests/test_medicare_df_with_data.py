import pytest
import pandas as pd
import medicare_utils as med


class TestGetCohortGetVarsToload(object):
    @pytest.fixture
    def init(self):
        return {
            'gender': None,
            'ages': None,
            'races': None,
            'race_col': 'race',
            'buyin_val': None,
            'hmo_val': None,
            'keep_vars': []}

    @pytest.fixture
    def mdf(self, year, percent):
        return med.MedicareDF(percent, year)

    @pytest.fixture(params=['0001', '01', '05', '20', '100'])
    def percent(self, request):
        return request.param

    @pytest.fixture(params=[2005, 2012])
    def year(self, request):
        return request.param

    def add_ehic(self, x, year):
        if year >= 2006:
            return x
        else:
            x.append('ehic')
            return x

    def assert_exp(self, mdf, init, exp, year):
        res = mdf._get_cohort_get_vars_toload(**init)
        exp = TestGetCohortGetVarsToload().add_ehic(exp, year)
        assert set(res[year]) == set(exp)

    # Only need to adjust these inputs
    @pytest.mark.parametrize(
    'inputs,extra_vars',
    [
    ({'gender': '1'}, ['sex']),
    ({'ages': range(70, 80)}, ['age']),
    ({'races': ['1'], 'race_col': 'race'}, ['race']),
    ({'races': ['1'], 'race_col': 'rti_race_cd'}, ['rti_race_cd']),
    ({'buyin_val': ['1', '2']}, ['buyin01', 'buyin02', 'buyin03', 'buyin04', 'buyin05', 'buyin06', 'buyin07', 'buyin08', 'buyin09', 'buyin10', 'buyin11', 'buyin12']),
    ({'hmo_val': ['1', '2']}, ['hmoind01', 'hmoind02', 'hmoind03', 'hmoind04', 'hmoind05', 'hmoind06', 'hmoind07', 'hmoind08', 'hmoind09', 'hmoind10', 'hmoind11', 'hmoind12']),
    ]) # yapf: disable
    def test_gender(self, year, mdf, init, inputs, extra_vars):
        for key, val in inputs.items():
            init[key] = val
        exp = ['bene_id']
        exp.extend(extra_vars)
        TestGetCohortGetVarsToload().assert_exp(mdf, init, exp, year)


class TestGetCohortExtractEachYear(object):
    """Tests for a single year of cohort extraction
    """

    @pytest.fixture
    def init(self):
        return {
            'gender': None,
            'ages': None,
            'races': None,
            'rti_race': False,
            'buyin_val': None,
            'hmo_val': None,
            'join': 'outer',
            'keep_vars': [],
            'dask': False,
            'verbose': False}

    @pytest.fixture
    def full_df(self):
        path = med.fpath(percent='0001', year=2012, data_type='bsfab')
        cols = [
            'bene_id', 'age', 'sex', 'race', 'rti_race_cd', 'buyin01',
            'buyin02', 'buyin03', 'buyin04', 'buyin05', 'buyin06', 'buyin07',
            'buyin08', 'buyin09', 'buyin10', 'buyin11', 'buyin12', 'hmoind01',
            'hmoind02', 'hmoind03', 'hmoind04', 'hmoind05', 'hmoind06',
            'hmoind07', 'hmoind08', 'hmoind09', 'hmoind10', 'hmoind11',
            'hmoind12']
        full_df = pd.read_parquet(path, columns=cols)
        print('Finished reading 0.01% bsfab data in 2012')
        return full_df

    # gender = 'm'
    # ages = None
    # races = None
    # rti_race = False
    # buyin_val = None
    # hmo_val = None
    # join = 'outer'
    # keep_vars = []
    # dask = False
    # verbose = False

    def setup_df(
            self,
            gender,
            ages,
            races,
            rti_race,
            buyin_val,
            hmo_val,
            join,
            keep_vars,
            dask,
            verbose,
            year_type='calendar',
            pct='0001',
            year=2012):
        """Set up to run _get_cohort_extract_each_year()

        Replicate get_cohort methods up to call of _get_cohort_extract_each_year
        """

        mdf = med.MedicareDF(pct, year, year_type=year_type)
        objs = mdf._get_cohort_type_check(
            gender=gender,
            ages=ages,
            races=races,
            rti_race=rti_race,
            buyin_val=buyin_val,
            hmo_val=hmo_val,
            join=join,
            keep_vars=keep_vars,
            dask=dask,
            verbose=verbose)
        gender = objs.gender
        ages = objs.ages
        races = objs.races
        rti_race = objs.rti_race
        race_col = objs.race_col
        buyin_val = objs.buyin_val
        hmo_val = objs.hmo_val
        join = objs.join
        keep_vars = objs.keep_vars
        dask = objs.dask
        verbose = objs.verbose

        toload_vars = mdf._get_cohort_get_vars_toload(
            gender, ages, races, race_col, buyin_val, hmo_val, keep_vars)

        return mdf, {
            'year': year,
            'toload_vars': toload_vars[year],
            'nobs_dropped': {
                year: {}},
            'gender': gender,
            'ages': ages,
            'races': races,
            'race_col': race_col,
            'buyin_val': buyin_val,
            'hmo_val': hmo_val,
            'join': join,
            'keep_vars': keep_vars,
            'dask': dask,
            'verbose': verbose}

    @pytest.mark.parametrize(
        'attrs,values,exp_vars,exp_isin_vals',
        [
        (
            ['gender'],
            ['m'],
            ['sex'],
            [['1']]
        ),
        (
            ['gender'],
            ['f'],
            ['sex'],
            [['2']]
        ),
        (
            ['gender'],
            [None],
            ['sex'],
            [['0', '1', '2']]
        ),
        (
            ['ages'],
            [range(75, 85)],
            ['age'],
            [range(75, 85)]
        ),
        (
            ['ages'],
            [[75, 76, 77, 78, 79, 80, 81, 82, 83, 84]],
            ['age'],
            [range(75, 85)]
        ),
        (
            ['races', 'rti_race'],
            ['white', False],
            ['race'],
            [['1']]
        ),
        (
            ['races', 'rti_race'],
            ['black', False],
            ['race'],
            [['2']]
        ),
        (
            ['races', 'rti_race'],
            ['asian', False],
            ['race'],
            [['4']]
        ),
        (
            ['races', 'rti_race'],
            [['white', 'black', 'asian'], False],
            ['race'],
            [['1', '2', '4']]
        ),
        (
            ['races', 'rti_race'],
            ['white', True],
            ['rti_race_cd'],
            [['1']]
        ),
        (
            ['races', 'rti_race'],
            ['black', True],
            ['rti_race_cd'],
            [['2']]
        ),
        (
            ['races', 'rti_race'],
            ['asian', True],
            ['rti_race_cd'],
            [['4']]
        ),
        (
            ['races', 'rti_race'],
            [['white', 'black', 'asian'], True],
            ['rti_race_cd'],
            [['1', '2', '4']]
        ),
        (
            ['buyin_val'],
            ['1'],
            ['buyin'],
            [['1']]
        ),
        (
            ['buyin_val'],
            [['1', '2', '3']],
            ['buyin'],
            [['1', '2', '3']]
        ),
        (
            ['buyin_val'],
            [['2', '3', 'B', 'C']],
            ['buyin'],
            [['2', '3', 'B', 'C']]
        ),
        (
            ['buyin_val'],
            [['3', 'C']],
            ['buyin'],
            [['3', 'C']]
        ),
        (
            ['hmo_val'],
            ['1'],
            ['hmoind'],
            [['1']]
        ),
        (
            ['hmo_val'],
            [['1', '2', '3']],
            ['hmoind'],
            [['1', '2', '3']]
        ),
        (
            ['hmo_val'],
            [['2', '3', 'B', 'C']],
            ['hmoind'],
            [['2', '3', 'B', 'C']]
        ),
        (
            ['hmo_val'],
            [['3', 'C']],
            ['hmoind'],
            [['3', 'C']]
        ),
        (
            ['gender', 'ages', 'races', 'rti_race', 'buyin_val'],
            ['m', range(67, 74), ['black', 'asian'], False, ['3', 'C']],
            ['sex', 'age', 'race', 'buyin'],
            [['1'], range(67, 74), ['2', '4'], ['3', 'C']]
        ),
        (
            ['gender', 'ages', 'races', 'rti_race', 'buyin_val'],
            ['f', range(67, 85), ['white', 'hispanic'], True, ['3', 'C']],
            ['sex', 'age', 'rti_race_cd', 'buyin'],
            [['2'], range(67, 85), ['1', '5'], ['3', 'C']]
        ),
        ]) # yapf: disable
    def test_df_is_expected(
            self, init, full_df, attrs, values, exp_vars, exp_isin_vals):
        for attr, value in zip(attrs, values):
            init[attr] = value

        mdf, attrs = TestGetCohortExtractEachYear().setup_df(**init)
        pl, nobs_dropped = mdf._get_cohort_extract_each_year(**attrs)
        pl = pl.index

        query = []
        for exp_var, exp_isin_val in zip(exp_vars, exp_isin_vals):
            if isinstance(exp_isin_val, range):
                exp_isin_val = list(exp_isin_val)
            if exp_var in ['buyin', 'hmoind']:
                for i in range(1, 13):
                    j = str(i).zfill(2)
                    query.append(f'{exp_var}{j}.isin({exp_isin_val})')
            else:
                query.append(f'{exp_var}.isin({exp_isin_val})')

        query = ' & '.join(query)
        expected = full_df.query(query)['bene_id']

        expected = pd.Index(expected.sort_values())
        pl = pd.Index(pl.sort_values())

        assert expected.equals(pl)
