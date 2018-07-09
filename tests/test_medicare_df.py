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
            rti_race=None,
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
                rti_race=None,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

    @pytest.mark.parametrize('gender', ['ma', 'mal', 'fem', 'femal', '3', '-1', 'unkn'])
    def test_gender_value_error(self, gender):
        mdf = med.MedicareDF('01', 2012)
        with pytest.raises(ValueError):
            mdf._get_cohort_type_check(
                gender=gender,
                ages=None,
                races=None,
                rti_race=None,
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
                rti_race=None,
                buyin_val=None,
                hmo_val=None,
                join='left',
                keep_vars=None,
                dask=True,
                verbose=True)

# gender
# ages
# races
# rti_race
# buyin_val
# hmo_val
# join='left'
# keep_vars
# dask
# verbose
