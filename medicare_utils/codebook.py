import json
import pkg_resources as pkg


def codebook(data_type):
    """Load variable codebook

    Args:
        data_type (str):

            Type of file to get codebook for

            - ``bsfab`` (`Beneficiary Summary File, Base segment`_)
            - ``med``   (`MedPAR File`_)
            - ``opc``   (`Outpatient File, Claims segment`_)

            .. _`Beneficiary Summary File, Base segment`: https://kylebarron.github.io/medicare-documentation/resdac/mbsf/#base-abcd-segment_2
            .. _`MedPAR File`: https://kylebarron.github.io/medicare-documentation/resdac/medpar-rif/#medpar-rif_1
            .. _`Outpatient File, Claims segment`: https://kylebarron.github.io/medicare-documentation/resdac/op-rif/#outpatient-rif_1

    Returns:
        ``dict`` with variable names as keys; values are another ``dict``. This
        inner ``dict`` has two keys: ``name``, where the value is the
        descriptive name of the variable, and ``values``, which is itself a
        ``dict`` with variable values as keys and value descriptions as values
        of the ``dict``.
    Examples:
        To get the labels of the values of ``clm_type``, in the ``med`` file,
        you could do

        .. code-block:: python

            >>> import medicare_utils as med
            >>> cbk = med.codebook('med')['clm_type']['values']

        Now ``cbk`` is a ``dict`` where the keys of the ``dict`` are the values
        the variable can take, and the values of the ``dict`` are the labels of
        the variable's values.

        .. code-block:: python

            >>> from pprint import pprint
            >>> pprint(cbk)
            {'10': 'HHA claim',
             '20': 'Non swing bed SNF claim',
             '30': 'Swing bed SNF claim',
             '40': 'Outpatient claim',
             '50': 'Hospice claim',
             '60': 'Inpatient claim',
             '61': "Inpatient 'Full-Encounter' claim",
             '62': 'Medicare Advantage IME/GME claims',
             '63': 'Medicare Advantage (no-pay) claims',
             '64': 'Medicare Advantage (paid as FFS) claim',
             '71': 'RIC O local carrier non-DMEPOS claim',
             '72': 'RIC O local carrier DMEPOS claim',
             '81': 'RIC M DMERC non-DMEPOS claim',
             '82': 'RIC M DMERC DMEPOS claim'}
    """
    path = pkg.resource_filename(
        'medicare_utils', f'data/codebook/{data_type}.json')

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise NotImplementedError(f'Haven\'t added {data_type} codebook yet')

    return data
