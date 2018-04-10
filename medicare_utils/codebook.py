import json
import pkg_resources as pkg


def codebook(data_type):
    path = pkg.resource_filename(
        'medicare_utils', f'data/codebook/{data_type}.json')

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise NotImplementedError(f'Haven\'t added {data_type} codebook yet')

    return data
