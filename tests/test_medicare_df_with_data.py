import pytest
import pandas as pd
import medicare_utils as med

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
        path = med.fpath(percent='01', year=2012, data_type='bsfab')
        full_df = pd.read_parquet(path)
        print('Finished reading 1% bsfab data in 2012')
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
            self, gender, ages, races, rti_race, buyin_val, hmo_val, join,
            keep_vars, dask, verbose, pct='01', year=2012):
        """Set up to run _get_cohort_extract_each_year()

        Replicate get_cohort methods up to call of _get_cohort_extract_each_year
        """

        mdf = med.MedicareDF(pct, year)
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
            'nobs_dropped': {year: {}},
            'gender': gender,
            'ages': ages,
            'races': races,
            'rti_race': rti_race,
            'race_col': race_col,
            'buyin_val': buyin_val,
            'hmo_val': hmo_val,
            'join': join,
            'keep_vars': keep_vars,
            'dask': dask,
            'verbose': verbose}


    # yapf: disable
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
        ])
    # yapf: enable
    def test_df_is_subset_of_expected(self, init, full_df, attrs, values, exp_vars, exp_isin_vals):
        for attr, value in zip(attrs, values):
            init[attr] = value

        mdf, attrs = TestGetCohortExtractEachYear().setup_df(**init)
        pl, nobs_dropped = mdf._get_cohort_extract_each_year(**attrs)
        pl = pl.index

        query = []
        for exp_var, exp_isin_val in zip(exp_vars, exp_isin_vals):
            if isinstance(exp_isin_val, range):
                exp_isin_val = list(exp_isin_val)
            query.append(f'{exp_var}.isin({exp_isin_val})')

        query = ' & '.join(query)
        expected = full_df.query(query)['bene_id']

        expected = pd.Index(expected.sort_values())
        pl = pd.Index(pl.sort_values())

        assert expected.equals(pl)
