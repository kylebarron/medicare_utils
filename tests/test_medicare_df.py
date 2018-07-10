import re
import pytest
import medicare_utils as med


class TestInit(object):
    @pytest.mark.parametrize(
        'pct,pct_act', [('0001', '0001'), ('01', '01'), ('05', '05'),
                        ('20', '20'), ('100', '100'), (0.01, '0001'), (1, '01'),
                        (5, '05'), (20, '20'), (100, '100')])
    def test_percents(self, pct, pct_act):
        mdf = med.MedicareDF(pct, 2012)
        assert mdf.percent == pct_act

    @pytest.mark.parametrize('pct', ['02', '45', 2, 56])
    def test_invalid_percents(self, pct):
        with pytest.raises(ValueError):
            med.MedicareDF(pct, 2012)

    @pytest.mark.parametrize(
        'years,years_act',
        [(2005, [2005]), (range(2010, 2013), range(2010, 2013)),
         ([2010, 2011, 2012], [2010, 2011, 2012])])
    def test_years(self, years, years_act):
        mdf = med.MedicareDF('01', years)
        assert mdf.years == years_act

    @pytest.mark.parametrize('years', ['2012', 2012.0])
    def test_invalid_years(self, years):
        with pytest.raises(TypeError):
            med.MedicareDF('01', years)

    @pytest.mark.parametrize('year_type', ['calendar', 'age'])
    def test_year_type(self, year_type):
        mdf = med.MedicareDF('01', [2011, 2012], year_type=year_type)
        mdf.year_type == year_type

    def test_age_year_type_with_one_year(self):
        with pytest.raises(ValueError):
            med.MedicareDF('01', 2012, year_type='age')
            med.MedicareDF('01', [2012], year_type='age')
            med.MedicareDF('01', range(2012, 2013), year_type='age')


class TestGetCohortTypeCheck(object):
    @pytest.mark.parametrize(
        'gender,expected',
        [(None, None), ('unknown', '0'), ('male', '1'), ('female', '2'),
         ('u', '0'), ('m', '1'), ('f', '2'), ('UNKNOWN', '0'), ('MALE', '1'),
         ('FEMALE', '2'), ('U', '0'), ('M', '1'), ('F', '2'), ('0', '0'),
         ('1', '1'), ('2', '2')])
    def test_gender(self, gender, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=gender,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=None,
            hmo_val=None,
            join='left',
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.gender == expected

    @pytest.mark.parametrize('gender', [['string_in_list'], [1], 1, 2, 0.1])
    def test_gender_type_error(self, gender):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(
                gender=gender,
                ages=None,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

    @pytest.mark.parametrize(
        'gender', ['ma', 'mal', 'fem', 'femal', '3', '-1', 'unkn'])
    def test_gender_value_error(self, gender):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(ValueError):
            mdf._get_cohort_type_check(
                gender=gender,
                ages=None,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

    @pytest.mark.parametrize('ages', ['65', 65.5, ['65'], [65, '66'], True])
    def test_ages_type_error(self, ages):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(
                gender=None,
                ages=ages,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

    @pytest.mark.parametrize('rti_race', ['1', 1])
    def test_rti_race(self, rti_race):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(
                gender=None,
                ages=None,
                races=None,
                rti_race=rti_race,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

    @pytest.mark.parametrize(
        'races,expected',
        [(None, None), ('unknown', ['0']), ('white', ['1']),
         ('black (or african-american)', ['2']), ('black', ['2']),
         ('african-american', ['2']), ('other', ['3']),
         ('asian pacific islander', ['4']), ('asian', ['4']),
         ('hispanic', ['5']), ('american indian alaska native', ['6']),
         ('american indian', ['6']), ('UNKNOWN', ['0']), ('WHITE', ['1']),
         ('BLACK (OR AFRICAN-AMERICAN)', ['2']), ('BLACK', ['2']),
         ('AFRICAN-AMERICAN', ['2']), ('OTHER', ['3']),
         ('ASIAN PACIFIC ISLANDER', ['4']), ('ASIAN', ['4']),
         ('HISPANIC', ['5']), ('AMERICAN INDIAN ALASKA NATIVE', ['6']),
         ('AMERICAN INDIAN', ['6']), (['white', 'black'], ['1', '2']),
         (['white', 'black', 'asian'], ['1', '2', '4']),
         (['white', 'asian'], ['1', '4']), (['black', 'asian'], ['2', '4']),
         (['0', '1', '2'], ['0', '1', '2']), ('0', ['0']), ('1', ['1']),
         ('2', ['2']), ('3', ['3']), ('4', ['4']), ('5', ['5']), ('6', ['6'])])
    def test_races_rti_true(self, races, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=races,
            rti_race=True,
            buyin_val=None,
            hmo_val=None,
            join='left',
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.races == expected

    @pytest.mark.parametrize(
        'races,expected',
        [(None, None), ('unknown', ['0']), ('white', ['1']), ('black', ['2']),
         ('other', ['3']), ('asian', ['4']), ('hispanic', ['5']),
         ('north american native', ['6']), ('UNKNOWN', ['0']), ('WHITE', ['1']),
         ('BLACK', ['2']), ('OTHER', ['3']), ('ASIAN', ['4']),
         ('HISPANIC', ['5']), ('NORTH AMERICAN NATIVE', ['6']),
         (['white', 'black'], ['1', '2']),
         (['white', 'black', 'asian'], ['1', '2', '4']),
         (['white', 'asian'], ['1', '4']), (['black', 'asian'], ['2', '4']),
         (['0', '1', '2'], ['0', '1', '2']), ('0', ['0']), ('1', ['1']),
         ('2', ['2']), ('3', ['3']), ('4', ['4']), ('5', ['5']), ('6', ['6'])])
    def test_races_rti_false(self, races, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=races,
            rti_race=False,
            buyin_val=None,
            hmo_val=None,
            join='left',
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.races == expected

    @pytest.mark.parametrize('buyin_val,expected', [
    ('3', ['3']),
    (['3'], ['3'])
    ])
    def test_buyin_val(self, buyin_val, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=buyin_val,
            hmo_val=None,
            join='left',
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.buyin_val == expected

    @pytest.mark.parametrize('hmo_val,expected', [
    ('3', ['3']),
    (['3'], ['3'])
    ])
    def test_hmo_val(self, hmo_val, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=None,
            hmo_val=hmo_val,
            join='left',
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.hmo_val == expected

    @pytest.mark.parametrize('keep_vars,expected', [
    ('3', ['3']),
    (['3'], ['3'])
    ])
    def test_keep_vars(self, keep_vars, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=None,
            hmo_val=None,
            join='left',
            keep_vars=keep_vars,
            dask=True,
            verbose=True)
        assert result.keep_vars == expected

    @pytest.mark.parametrize('allowed_join,expected', [
    ('left', 'left'),
    ('right', 'right'),
    ('inner', 'inner'),
    ('outer', 'outer')
    ])
    def test_allowed_join(self, allowed_join, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._get_cohort_type_check(
            gender=None,
            ages=None,
            races=None,
            rti_race=False,
            buyin_val=None,
            hmo_val=None,
            join=allowed_join,
            keep_vars=None,
            dask=True,
            verbose=True)
        assert result.join == expected

    def test_allowed_join_value_error(self):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(ValueError):
            mdf._get_cohort_type_check(
                gender=None,
                ages=None,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='other',
                keep_vars=None,
                dask=True,
                verbose=True)

    def test_dask_type_error(self):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(
                gender=None,
                ages=None,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask='string',
                verbose=True)

    def test_verbose_type_error(self):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_cohort_type_check(
                gender=None,
                ages=None,
                races=None,
                rti_race=False,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose='string')

class TestGetPattern(object):
    def test_get_pattern_str(self):
        mdf = med.MedicareDF('01', 2012)
        assert mdf._get_pattern('string') == 'string'

    def test_get_pattern_regex(self):
        mdf = med.MedicareDF('01', 2012)
        regex = re.compile('regex_match')
        assert mdf._get_pattern(regex) == 'regex_match'

    @pytest.mark.parametrize('obj', [True, 1, 1.0, ['string'], [re.compile('regex')]])
    def test_get_pattern_invalid_type(self, obj):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(TypeError):
            mdf._get_pattern(obj)




class TestCreateRenameDict(object):
    @pytest.mark.parametrize('hcpcs,icd9_dx,icd9_sg,rename,expected', [
        (None, None, None, {}, {}),
        # ('a', 'b', 'c', {'hcpcs': '1', 'icd9_dx': '2', 'icd9_sg': '3'}, {'a': '1', 'b': '2', 'c': '3'}),
        ('a', 'b', 'c', {'hcpcs': ['1'], 'icd9_dx': ['2'], 'icd9_sg': ['3']}, {'a': '1', 'b': '2', 'c': '3'}),
        (['a'], ['b'], ['c'], {'hcpcs': ['1'], 'icd9_dx': ['2'], 'icd9_sg': ['3']}, {'a': '1', 'b': '2', 'c': '3'}),
        (['a'], ['b'], ['c'], {'hcpcs': {'a': '1'}, 'icd9_dx': {'b': '2'}, 'icd9_sg': {'c': '3'}}, {'a': '1', 'b': '2', 'c': '3'}),
        (['a', 'd'], ['b', 'e'], ['c', 'f'],
         {'hcpcs': ['1', '4'], 'icd9_dx': ['2', '5'], 'icd9_sg': ['3', '6']},
         {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6'})

    ])
    def test_hcpcs_list_rename(self, hcpcs, icd9_dx, icd9_sg, rename, expected):
        mdf = med.MedicareDF('01', 2012)
        result = mdf._create_rename_dict(hcpcs=hcpcs, icd9_dx=icd9_dx, icd9_sg=icd9_sg, rename=rename)
        assert result == expected
















# verbose
