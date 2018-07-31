import re
import pytest
import pandas as pd
import medicare_utils as med


class TestInit(object):
    # All the non-default arguments
    @pytest.fixture
    def init(self):
        return {'percent': '01', 'years': 2012}

    @pytest.mark.parametrize(
        'pct,pct_act',
        [('0001', '0001'),
         ('01', '01'),
         ('05', '05'),
         ('20', '20'),
         ('100', '100'),
         (0.01, '0001'),
         (1, '01'),
         (5, '05'),
         (20, '20'),
         (100, '100')]) # yapf: disable
    def test_percents(self, init, pct, pct_act):
        init['percent'] = pct
        mdf = med.MedicareDF(**init)
        assert mdf.percent == pct_act

    @pytest.mark.parametrize('pct', ['02', '45', 2, 56])
    def test_invalid_percents(self, init, pct):
        init['percent'] = pct
        with pytest.raises(ValueError):
            med.MedicareDF(**init)

    @pytest.mark.parametrize(
        'years,years_act',
        [(2005, [2005]),
         (range(2010, 2013), range(2010, 2013)),
         ([2010, 2011, 2012], [2010, 2011, 2012])]) # yapf: disable
    def test_years(self, init, years, years_act):
        init['years'] = years
        mdf = med.MedicareDF(**init)
        assert mdf.years == years_act

    @pytest.mark.parametrize('years', ['2012', 2012.0])
    def test_invalid_years(self, init, years):
        init['years'] = years
        with pytest.raises(TypeError):
            med.MedicareDF(**init)

    @pytest.mark.parametrize('year_type', ['calendar', 'age'])
    def test_year_type(self, year_type):
        mdf = med.MedicareDF('01', [2011, 2012], year_type=year_type)
        mdf.year_type == year_type

    @pytest.mark.parametrize(
        'years', [2012, [2012], range(2012, 2013), [2010, 2012]])
    def test_invalid_age_years(self, init, years):
        init['year_type'] = 'age'
        init['years'] = years
        with pytest.raises(ValueError):
            med.MedicareDF(**init)


class TestGetCohortTypeCheck(object):
    @pytest.fixture
    def init(self):
        return {
            'gender': None,
            'ages': None,
            'races': None,
            'rti_race': False,
            'buyin_val': None,
            'hmo_val': None,
            'join': 'left',
            'keep_vars': None,
            'dask': False,
            'verbose': True}

    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', 2012)

    @pytest.mark.parametrize(
        'gender,expected',
        [(None, None),
         ('unknown', '0'),
         ('male', '1'),
         ('female', '2'),
         ('u', '0'),
         ('m', '1'),
         ('f', '2'),
         ('UNKNOWN', '0'),
         ('MALE', '1'),
         ('FEMALE', '2'),
         ('U', '0'),
         ('M', '1'),
         ('F', '2'),
         ('0', '0'),
         ('1', '1'),
         ('2', '2')]) # yapf: disable
    def test_gender(self, mdf, init, gender, expected):
        init['gender'] = gender
        result = mdf._get_cohort_type_check(**init)
        assert result.gender == expected

    @pytest.mark.parametrize(
        'gender,error', [
            (['string_in_list'], TypeError),
            ([1], TypeError),
            (1, TypeError),
            (2, TypeError),
            (0.1, TypeError),
            ('ma', ValueError),
            ('mal', ValueError),
            ('fem', ValueError),
            ('femal', ValueError),
            ('3', ValueError),
            ('-1', ValueError),
            ('unkn', ValueError), ])
    def test_gender_type_error(self, mdf, init, gender, error):
        init['gender'] = gender
        with pytest.raises(error):
            mdf._get_cohort_type_check(**init)

    @pytest.mark.parametrize('ages', ['65', 65.5, ['65'], [65, '66'], True])
    def test_ages_type_error(self, mdf, init, ages):
        init['ages'] = ages
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(**init)

    @pytest.mark.parametrize('rti_race', ['1', '0', 0, 1])
    def test_rti_race(self, mdf, init, rti_race):
        init['rti_race'] = rti_race
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(**init)

    @pytest.mark.parametrize(
        'races,expected',
        [(None, None),
         ('unknown', ['0']),
         ('white', ['1']),
         ('black (or african-american)', ['2']),
         ('black', ['2']),
         ('african-american', ['2']),
         ('other', ['3']),
         ('asian pacific islander', ['4']),
         ('asian', ['4']),
         ('hispanic', ['5']),
         ('american indian alaska native', ['6']),
         ('american indian', ['6']),
         ('UNKNOWN', ['0']),
         ('WHITE', ['1']),
         ('BLACK (OR AFRICAN-AMERICAN)', ['2']),
         ('BLACK', ['2']),
         ('AFRICAN-AMERICAN', ['2']),
         ('OTHER', ['3']),
         ('ASIAN PACIFIC ISLANDER', ['4']),
         ('ASIAN', ['4']),
         ('HISPANIC', ['5']),
         ('AMERICAN INDIAN ALASKA NATIVE', ['6']),
         ('AMERICAN INDIAN', ['6']),
         (['white', 'black'], ['1', '2']),
         (['white', 'black', 'asian'], ['1', '2', '4']),
         (['white', 'asian'], ['1', '4']),
         (['black', 'asian'], ['2', '4']),
         (['0', '1', '2'], ['0', '1', '2']),
         ('0', ['0']),
         ('1', ['1']),
         ('2', ['2']),
         ('3', ['3']),
         ('4', ['4']),
         ('5', ['5']),
         ('6', ['6'])]) # yapf: disable
    def test_races_rti_true(self, mdf, init, races, expected):
        init['rti_race'] = True
        init['races'] = races
        result = mdf._get_cohort_type_check(**init)
        assert result.races == expected

    @pytest.mark.parametrize(
        'races,expected',
        [(None, None),
         ('unknown', ['0']),
         ('white', ['1']),
         ('black', ['2']),
         ('other', ['3']),
         ('asian', ['4']),
         ('hispanic', ['5']),
         ('north american native', ['6']),
         ('UNKNOWN', ['0']),
         ('WHITE', ['1']),
         ('BLACK', ['2']),
         ('OTHER', ['3']),
         ('ASIAN', ['4']),
         ('HISPANIC', ['5']),
         ('NORTH AMERICAN NATIVE', ['6']),
         (['white', 'black'], ['1', '2']),
         (['white', 'black', 'asian'], ['1', '2', '4']),
         (['white', 'asian'], ['1', '4']),
         (['black', 'asian'], ['2', '4']),
         (['0', '1', '2'], ['0', '1', '2']),
         ('0', ['0']),
         ('1', ['1']),
         ('2', ['2']),
         ('3', ['3']),
         ('4', ['4']),
         ('5', ['5']),
         ('6', ['6'])
         ]) # yapf: disable
    def test_races_rti_false(self, mdf, init, races, expected):
        init['rti_race'] = False
        init['races'] = races
        result = mdf._get_cohort_type_check(**init)
        assert result.races == expected

    @pytest.mark.parametrize(
        'buyin_val,expected', [('3', ['3']), (['3'], ['3'])])
    def test_buyin_val(self, mdf, init, buyin_val, expected):
        init['buyin_val'] = buyin_val
        result = mdf._get_cohort_type_check(**init)
        assert result.buyin_val == expected

    @pytest.mark.parametrize('hmo_val,expected', [('3', ['3']), (['3'], ['3'])])
    def test_hmo_val(self, mdf, init, hmo_val, expected):
        init['hmo_val'] = hmo_val
        result = mdf._get_cohort_type_check(**init)
        assert result.hmo_val == expected

    @pytest.mark.parametrize(
        'keep_vars,expected',
        [
        ('3', ['3']),
        (['3'], ['3']),
        (['3', re.compile('a')], ['3', re.compile('a')]),
        (re.compile('a'), [re.compile('a')]),
        ]) # yapf: disable
    def test_keep_vars(self, mdf, init, keep_vars, expected):
        init['keep_vars'] = keep_vars
        result = mdf._get_cohort_type_check(**init)
        assert result.keep_vars == expected

    @pytest.mark.parametrize('join,expected',
        [('left', 'left'),
         ('right', 'right'),
         ('inner', 'inner'),
         ('outer', 'outer')]) # yapf: disable
    def test_allowed_join(self, mdf, init, join, expected):
        init['join'] = join
        result = mdf._get_cohort_type_check(**init)
        assert result.join == expected

    def test_allowed_join_value_error(self, mdf, init):
        init['join'] = 'invalid_string'
        with pytest.raises(ValueError):
            mdf._get_cohort_type_check(**init)

    def test_dask_type_error(self, mdf, init):
        init['dask'] = 'string'
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(**init)

    def test_verbose_type_error(self, mdf, init):
        init['verbose'] = 'string'
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(**init)


class TestGetCohortMonthFilter(object):
    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', [2010, 2011, 2012], year_type='age')

    @pytest.fixture
    def pl(self):
        # yapf: disable
        data = [
            [1, '2','2','1','1','2','2','1','2','1','2','2','2'],
            [2, '2','2','2','1','2','2','1','1','2','2','2','1'],
            [3, '2','2','2','1','2','2','1','1','1','2','2','2'],
            [4, '1','2','1','1','1','1','2','2','2','1','2','2'],
            [5, '2','2','2','1','1','2','2','1','2','1','2','1'],
            [6, '2','1','1','1','2','1','1','1','1','2','2','2'],
            [7, '2','2','1','1','2','2','1','2','1','2','2','2'],
            [8, '2','2','2','1','2','2','1','1','2','2','2','2'],
            [9, '2','2','1','1','2','2','1','1','2','2','2','2'],
            [10, '1','2','1','1','1','1','2','2','2','2','2','2'],
            [11, '2','2','2','1','1','2','2','1','2','2','2','2'],
            [12, '2','1','1','1','2','1','1','1','1','2','2','2']]
        # yapf: enable
        cols = [
            'dob_month', 'var01', 'var02', 'var03', 'var04', 'var05', 'var06',
            'var07', 'var08', 'var09', 'var10', 'var11', 'var12']
        return pd.DataFrame.from_records(data, columns=cols)

    @pytest.fixture
    def exp(self):
        return pd.DataFrame({
            'dob_month': [1, 2, 3, 9, 10, 11, 12],
            'var_younger': [True, True, True, False, False, False, False],
            'var_older': [False, False, False, True, True, True, True]},
                            index=[0, 1, 2, 8, 9, 10, 11])

    def test_month_filter_mid(self, mdf, pl, exp):
        df = mdf._get_cohort_month_filter(
            pl=pl, var='var', values=['2'], year=2011, keep_vars=[])
        assert df.equals(exp)

    def test_month_filter_first(self, mdf, pl, exp):
        df = mdf._get_cohort_month_filter(
            pl=pl, var='var', values=['2'], year=2010, keep_vars=[])
        exp = exp.loc[exp['var_older'], ['dob_month', 'var_older']]
        assert df.equals(exp)

    def test_month_filter_last(self, mdf, pl, exp):
        df = mdf._get_cohort_month_filter(
            pl=pl, var='var', values=['2'], year=2012, keep_vars=[])
        exp = exp.loc[exp['var_younger'], ['dob_month', 'var_younger']]
        assert df.equals(exp)


class TestStrInKeepVars(object):
    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', 2012)

    @pytest.mark.parametrize(
        'instr,keep_vars,res',
        [('a', ['a', 'b', 'c'], True),
        ('d', ['a', 'b', 'c'], False),
        ('a', ['a', re.compile(r'b')], True),
        ('d', ['a', re.compile(r'b')], False),
        ('a', [re.compile(r'a')], True),
        ('a', [re.compile(r'b')], False)]) # yapf: disable
    def test_str_in_keep_vars(self, mdf, instr, keep_vars, res):
        assert res == mdf._str_in_keep_vars(instr, keep_vars)


class TestGetPattern(object):
    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', 2012)

    def test_get_pattern_str(self, mdf):
        assert mdf._get_pattern('string') == 'string'

    def test_get_pattern_regex(self, mdf):
        regex = re.compile('regex_match')
        assert mdf._get_pattern(regex) == 'regex_match'

    @pytest.mark.parametrize(
        'obj', [True, 1, 1.0, ['string'], [re.compile('regex')]])
    def test_get_pattern_invalid_type(self, mdf, obj):
        with pytest.raises(TypeError):
            mdf._get_pattern(obj)


class TestCreateRenameDict(object):
    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', 2012)

    @pytest.mark.parametrize('hcpcs,icd9_dx,icd9_sg,rename,expected', [
        (None, None, None, {}, {}),
        ('a', 'b', 'c',
            {'hcpcs': '1', 'icd9_dx': '2', 'icd9_sg': '3'},
            {'a': '1', 'b': '2', 'c': '3'}),
        ('a', 'b', 'c',
            {'hcpcs': ['1'], 'icd9_dx': ['2'], 'icd9_sg': ['3']},
            {'a': '1', 'b': '2', 'c': '3'}),
        (['a'], ['b'], ['c'],
            {'hcpcs': ['1'], 'icd9_dx': ['2'], 'icd9_sg': ['3']},
            {'a': '1', 'b': '2', 'c': '3'}),
        (['a'], ['b'], ['c'],
            {'hcpcs': {'a': '1'}, 'icd9_dx': {'b': '2'}, 'icd9_sg': {'c': '3'}},
            {'a': '1', 'b': '2', 'c': '3'}),
        (['a', 'd'], ['b', 'e'], ['c', 'f'],
             {'hcpcs': ['1', '4'], 'icd9_dx': ['2', '5'], 'icd9_sg': ['3', '6']},
             {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6'}),
        (['a', 'b'], ['c', 'd'], ['e', 'f'],
            {'hcpcs': {'a': '1'}, 'icd9_dx': {'c': '2'}, 'icd9_sg': {'e': '3'}},
            {'a': '1', 'c': '2', 'e': '3'}),
    ]) # yapf: disable
    def test_rename_dict_noerror(
            self, mdf, hcpcs, icd9_dx, icd9_sg, rename, expected):
        codes = {'hcpcs': hcpcs, 'icd9_dx': icd9_dx, 'icd9_sg': icd9_sg}
        result = mdf._create_rename_dict(codes=codes, rename=rename)
        assert result == expected

    @pytest.mark.parametrize('hcpcs,icd9_dx,icd9_sg,rename', [
        (None, None, None,
            {'hcpcs': ['1', '2'],
             'icd9_dx': ['2', '3'],
             'icd9_sg': ['3', '4']}),
        ('a', 'b', 'c',
            {'hcpcs': ['1', '2'],
             'icd9_dx': ['2', '3'],
             'icd9_sg': ['3', '4']}),
        ('a', 'b', 'c', {'hcpcs': [], 'icd9_dx': [], 'icd9_sg': []}),
        (['a', 'b'], ['c', 'd'], ['e', 'f'],
            {'hcpcs': [], 'icd9_dx': [], 'icd9_sg': []}),
        (['a', 'b'], ['c', 'd'], ['e', 'f'],
            {'hcpcs': '1', 'icd9_dx': '2', 'icd9_sg': '3'}),
    ]) # yapf: disable
    def test_rename_dict_wrong_list_len(
            self, mdf, hcpcs, icd9_dx, icd9_sg, rename):
        codes = {'hcpcs': hcpcs, 'icd9_dx': icd9_dx, 'icd9_sg': icd9_sg}
        with pytest.raises(AssertionError):
            mdf._create_rename_dict(codes=codes, rename=rename)

    @pytest.mark.parametrize('hcpcs,icd9_dx,icd9_sg,rename', [
        (None, None, None,
            {'hcpcs': '1', 'icd9_dx': '2', 'icd9_sg': '3'}),
        ('a', 'b', 'c',
            {'hcpcs': {'a': '1', 'x': '5'},
             'icd9_dx': {'b': '2', 'y': '6'},
             'icd9_sg': {'c': '3', 'z': '7'}}),
    ]) # yapf: disable
    def test_rename_dict_wrong_dict_length(
            self, mdf, hcpcs, icd9_dx, icd9_sg, rename):
        codes = {'hcpcs': hcpcs, 'icd9_dx': icd9_dx, 'icd9_sg': icd9_sg}
        with pytest.raises(AssertionError):
            mdf._create_rename_dict(codes=codes, rename=rename)


class TestSearchForCodesTypeCheck(object):
    @pytest.fixture
    def init(self):
        init = {
            'data_types': 'med',
            'hcpcs': None,
            'icd9_dx': None,
            'icd9_dx_max_cols': None,
            'icd9_sg': None,
            'keep_vars': {},
            'collapse_codes': True,
            'rename': {
                'hcpcs': None,
                'icd9_dx': None,
                'icd9_sg': None},
            'convert_ehic': True,
            'verbose': False}
        return init

    @pytest.fixture
    def mdf(self):
        return med.MedicareDF('01', 2012)

    @pytest.mark.parametrize(
        'data_types,expected',
        [('carc', ['carc']),
         (['carc'], ['carc']),
         (['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr'],
            ['carc', 'carl', 'ipc', 'ipr', 'med', 'opc', 'opr']),
        ]) # yapf: disable
    def test_data_types(self, mdf, init, data_types, expected):
        init['data_types'] = data_types
        result = mdf._search_for_codes_type_check(**init)
        assert result.data_types == expected

    @pytest.mark.parametrize(
        'data_types,error',
        [(None, TypeError),
         ('a', ValueError),
         ('sdb', ValueError),
         (1, TypeError)]) # yapf: disable
    def test_wrong_data_types(self, mdf, init, data_types, error):
        init['data_types'] = data_types
        with pytest.raises(error):
            mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'codes,error',
        [(1, TypeError),
         (1.1, TypeError),
         ([1], TypeError),
         ([1.1], TypeError),
         ([['a']], TypeError),
         (['a', ['b']], TypeError),
         ([[re.compile('a')]], TypeError),
         ([re.compile('a'), [re.compile('b')]], TypeError),
         ]) # yapf: disable
    def test_codes_error(self, mdf, init, codes, error):
        for x in ['hcpcs', 'icd9_dx', 'icd9_sg']:
            init[x] = codes
            with pytest.raises(error):
                mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'hcpcs,icd9_dx,icd9_sg,expected',
        [
        (None, None, None,
         {'hcpcs': [],
          'icd9_dx': [],
          'icd9_sg': []}),
        ('a', 'a', 'a',
         {'hcpcs': ['a'],
          'icd9_dx': ['a'],
          'icd9_sg': ['a']}),
        (['a'], ['a'], ['a'],
         {'hcpcs': ['a'],
          'icd9_dx': ['a'],
          'icd9_sg': ['a']}),
        ('a', 'b', 'c',
         {'hcpcs': ['a'],
          'icd9_dx': ['b'],
          'icd9_sg': ['c']}),
        (['a'], ['b'], ['c'],
         {'hcpcs': ['a'],
          'icd9_dx': ['b'],
          'icd9_sg': ['c']}),
        ('', '', '',
         {'hcpcs': [''],
          'icd9_dx': [''],
          'icd9_sg': ['']}),
        ([''], [''], [''],
         {'hcpcs': [''],
          'icd9_dx': [''],
          'icd9_sg': ['']}),
        ('a', re.compile('b'), 'c',
         {'hcpcs': ['a'],
          'icd9_dx': [re.compile('b')],
          'icd9_sg': ['c']}),
        (re.compile('a'), re.compile('a'), re.compile('a'),
         {'hcpcs': [re.compile('a')],
          'icd9_dx': [re.compile('a')],
          'icd9_sg': [re.compile('a')]}),
        ([re.compile('a')], [re.compile('a')], [re.compile('a')],
         {'hcpcs': [re.compile('a')],
          'icd9_dx': [re.compile('a')],
          'icd9_sg': [re.compile('a')]}),
         ]) # yapf: disable
    def test_codes(self, mdf, init, hcpcs, icd9_dx, icd9_sg, expected):
        init['collapse_codes'] = True
        init['hcpcs'] = hcpcs
        init['icd9_dx'] = icd9_dx
        init['icd9_sg'] = icd9_sg
        result = mdf._search_for_codes_type_check(**init)
        assert result.codes == expected

    @pytest.mark.parametrize(
        'hcpcs,icd9_dx,icd9_sg,error',
        [
        ('a', 'a', None, ValueError),
        (None, 'a', 'a', ValueError),
        ('a', None, 'a', ValueError),
        (re.compile('a'), re.compile('a'), None, ValueError),
        (None, re.compile('a'), re.compile('a'), ValueError),
        (re.compile('a'), None, re.compile('a'), ValueError),
        (re.compile('a'), 'a', None, ValueError),
        (None, re.compile('a'), 'a', ValueError),
        (re.compile('a'), None, 'a', ValueError),
        ]) # yapf: disable
    def test_dup_code_patterns(self, mdf, init, hcpcs, icd9_dx, icd9_sg, error):
        init['collapse_codes'] = False
        init['hcpcs'] = hcpcs
        init['icd9_dx'] = icd9_dx
        init['icd9_sg'] = icd9_sg
        with pytest.raises(error):
            mdf._search_for_codes_type_check(**init)

    def test_icd9_dx_max_cols(self, mdf, init):
        init['icd9_dx'] = None
        init['icd9_dx_max_cols'] = 5
        with pytest.raises(ValueError):
            mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'value,error',
        [(1, TypeError),
         ('a', TypeError),
         ([1], TypeError),
         (True, TypeError),
         ({'invalid_key': 'string'}, ValueError),
         ({'med': 1}, TypeError),
         ({'med': True}, TypeError),
         ]) # yapf: disable
    def test_keep_vars_error(self, mdf, init, value, error):
        init['keep_vars'] = value
        with pytest.raises(error):
            mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'value,expected',
        [({'med': 'string'}, {'med': ['string']}),
         ({'med': ['string']}, {'med': ['string']})]) # yapf: disable
    def test_keep_vars(self, mdf, init, value, expected):
        init['keep_vars'] = value
        result = mdf._search_for_codes_type_check(**init)
        assert result.keep_vars == expected

    @pytest.mark.parametrize(
        'hcpcs,icd9_dx,icd9_sg,rename',
        [(None, None, None, {
            'wrongkey': ['new_name']}),
         ('a', 'b', 'c', {
             'wrongkey': ['new_name']})])
    # More `rename` tests in TestCreateRenameDict class
    def test_rename_dict_wrong_dict_key(
            self, mdf, init, hcpcs, icd9_dx, icd9_sg, rename):
        init['hcpcs'] = hcpcs
        init['icd9_dx'] = icd9_dx
        init['icd9_sg'] = icd9_sg
        init['rename'] = rename
        with pytest.raises(ValueError):
            mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'rename,error',
        [({'hcpcs': ['somevalue']}, ValueError),
        ({'icd9_dx': 'string'}, ValueError)]) # yapf: disable
    # Rename argument not allowed when collapse_codes is True
    def test_rename_collapse_codes_error(self, mdf, init, rename, error):
        init['collapse_codes'] = True
        init['rename'] = rename
        with pytest.raises(error):
            mdf._search_for_codes_type_check(**init)

    @pytest.mark.parametrize(
        'value,var,error',
        [(1, 'collapse_codes', TypeError),
         ('a', 'collapse_codes', TypeError),
         ([True], 'collapse_codes', TypeError),
         (None, 'collapse_codes', TypeError),
         (1, 'convert_ehic', TypeError),
         ('a', 'convert_ehic', TypeError),
         ([True], 'convert_ehic', TypeError),
         (None, 'convert_ehic', TypeError),
         (1, 'verbose', TypeError),
         ('a', 'verbose', TypeError),
         ([True], 'verbose', TypeError),
         (None, 'verbose', TypeError),
         ]) # yapf: disable
    def test_bool_input_type_error(self, mdf, init, value, var, error):
        init[var] = value
        with pytest.raises(error):
            mdf._search_for_codes_type_check(**init)


# verbose
