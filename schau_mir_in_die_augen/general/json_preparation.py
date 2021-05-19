""" Prepare content to be json compatible """

import warnings


def clean_conten4json(some_content: any) -> any:
    """ Convert any content to be json compatible. """
    if isinstance(some_content, (str, int, float, bool)):
        return some_content
    if isinstance(some_content, dict):
        return clean_dict4json(some_content)
    if isinstance(some_content, list):
        return clean_list4json(some_content)
    if isinstance(some_content, tuple):
        warnings.warn('Tuple can not be stored in Python! Will be converted to list')
        return clean_list4json([entry for entry in some_content])
    raise Exception('This should not be reached')


def clean_dict4json(some_dict: dict) -> dict:
    """ Convert a dictionary to be json compatible. """
    kill_list = []
    for key in some_dict:
        if isinstance(key, (int, float, tuple)):
            kill_list.append(key)
        else:
            some_dict[key] = clean_conten4json(some_dict[key])

    for key in kill_list:
        warnings.warn('Fieldname "{}" ({}) will be converted to string!'.format(key, type(key)))
        some_dict[str(key)] = some_dict[key]
        del some_dict[key]

    return some_dict


def clean_list4json(some_list: [list, tuple]) -> [list, tuple]:
    """ Convert a list or a tuple to be json compatible. """
    kill_list = []
    for idd, entry in enumerate(some_list):
        if isinstance(entry, tuple):
            kill_list.append(idd)
        else:
            some_list[idd] = clean_conten4json(entry)

    for idd in reversed(kill_list):
        print(' Remove Entry: {}'.format(type(some_list[idd])))
        del some_list[idd]

    return some_list
