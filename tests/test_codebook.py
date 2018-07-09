import pytest
from medicare_utils import codebook

class TestCodebook(object):
    @pytest.fixture(params=['bsfab', 'med', 'opc'])
    def d(self, request):
        return codebook(request.param)

    def test_unique_varnames(self, d):
        varnames = [key for key, val in d.items()]
        assert len(varnames) == len(set(varnames))

    def test_dict_keys(self, d):
        varnames = [key for key, val in d.items()]
        for varname in varnames:
            keys = [key for key, val in d[varname].items()]
            assert len(keys) == 2
            assert 'name' in keys
            assert 'values' in keys
