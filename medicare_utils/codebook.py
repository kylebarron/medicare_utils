import json
import pkg_resources as pkg


def codebook(data_type):
    if data_type == 'med':
        path = pkg.resource_filename(
            'medicare_utils', 'data/codebook/medpar.json')

    with open(path) as f:
        data = json.load(f)

    return data
