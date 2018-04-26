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
        To get the labels of the values of ``clm_type``, in the ``bsfab`` file,
        you could do

        .. code-block:: python

            import medicare_utils as med
            cbk = med.codebook('bsfab')['clm_type']['values']

        Now ``cbk`` is a ``dict`` where the keys of the ``dict`` are the values
        the variable can take, and the values of the ``dict`` are the labels of
        the variable's values.
    """
    path = pkg.resource_filename(
        'medicare_utils', f'data/codebook/{data_type}.json')

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise NotImplementedError(f'Haven\'t added {data_type} codebook yet')

    return data
