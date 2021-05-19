import os
import warnings

from pathlib import Path

import pickle
import json

from schau_mir_in_die_augen.general.json_preparation import clean_conten4json

local_keywords: list = ['[smida]', '[model]', '[model_user]', '[bet]', '[bus]']


def save_parameter_file(file_name: str, parameter_dict: dict) -> str:
    """ Load content from a parameter file and returns dictionary.

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
        Filetype depends on file ending. Possible types: '.pickle'
    :param parameter_dict: dictionary with parameters

    :return: absolut path where the file was saved
    """

    assert isinstance(parameter_dict, dict)

    if file_name.endswith('.json'):
        return save_json_file(file_name, parameter_dict)
    elif file_name.endswith('.pickle'):
        return save_pickle_file(file_name, parameter_dict)
    else:
        warnings.warn('Unknown file ending! Using json to save.')
        return save_json_file(file_name, parameter_dict)


def load_parameter_file(file_name: str, expected_fields: list = None) -> dict:
    """ Load content from a parameter file and returns dictionary.

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    :param expected_fields: verify to get the correct dictionary or label content without dict
        This is done to keep it compatible to old pickle loading in gui_bokeh.

    :return: dictionary from saved parameter file
    """

    if file_name.endswith('.json'):
        content = load_json_file(file_name)
    elif file_name.endswith('.pickle'):
        content = load_pickle_file(file_name)
    else:
        warnings.warn('Unknown file ending! Try json to load.')
        content = load_json_file(file_name)

    if expected_fields is not None:
        if isinstance(content, dict):
            if list(content) != expected_fields:
                raise Exception('Expected fields: {}\nLoaded fields: {}'.format(expected_fields, list(content)))
        elif len(content) == len(expected_fields):
            warnings.warn('Loaded unlabeled content. Using expected_fields to name!')
            content = dict(zip(expected_fields, content))
        else:
            raise Exception(
                "Loaded content is no dict ({typ}) and doesn't match number of expected field ({fie} vs {con})".format(
                    typ=type(content), fie=len(expected_fields), con=len(content)))
    elif not isinstance(content, dict):
        raise Exception('Content of loaded File is no dict!')

    return content


def load_json_file(file_name: str):
    """ load content from a pickle file

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    :return: Will return saved filetype?
    """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'rb') as f:
            content = json.load(f)
    except Exception:
        print(f'Failed loading file: "{file_name}" in "{os.getcwd()}"')
        raise
    return content


def save_json_file(file_name: str, content) -> str:
    """ save content in a json file.

        :param file_name: name of file
            If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
        :param content: content to be saved
        :return: will return absolute filepath to saved file
        """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(clean_conten4json(content), f)
    except FileNotFoundError:
        print(f'There is no directory "{file_name}"')
    except Exception:
        print(f'Failed to save "{type(content)}" in file: "{file_name}" from "{os.getcwd()}"')
        raise

    return file_name


def load_pickle_file(file_name: str):
    """ load content from a pickle file.

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    :return: Will return saved filetype
    """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'rb') as f:
            content = pickle.load(f)
    except Exception:
        print(f'Failed loading file: "{file_name}" in "{os.getcwd()}"')
        raise
    return content


def save_pickle_file(file_name: str, content) -> str:
    """ save content in a pickle file

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    :param content: content to be saved
    :return: will return absolute filepath to saved file
    """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'wb') as f:
            pickle.dump(content, f)
    except FileNotFoundError:
        print(f'There is no directory "{file_name}"')
    except Exception:
        print(f'Failed to save file: "{file_name}" from "{os.getcwd()}"')
        raise

    return file_name


def replace_local_path(file_name: str) -> str:
    """ Will search for keyword at the beginning and replace it with actual filepath.

    If local folder is selected it will be created if not there.

    :param file_name:
        '[model]' will be replaced by path to model folder (with ending slash)
        '[bokeh_settings]' will be replaced by user folder
    :return file_name: with changed keyword, or like before
    """

    #   see local_keywords
    bokeh_repo = '[bet]'
    bokeh_user = '[bus]'
    smida = '[smida]'
    model_repo = '[model]'
    model_user = '[model_user]'

    if file_name is not None:

        if file_name.startswith(smida):
            file_name = create_top_level_path(file_name[len(smida):])

        elif file_name.startswith(bokeh_repo):
            file_name = create_top_level_path(file_name[len(bokeh_repo):], 'settings', 'bokeh')

        elif file_name.startswith(model_repo):
            file_name = create_top_level_path(file_name[len(model_repo):], 'model')

        elif file_name.startswith(bokeh_user):
            file_name = create_user_path(file_name[len(bokeh_user):], 'bokeh')

        elif file_name.startswith(model_user):
            file_name = create_user_path(file_name[len(model_user):], 'model')

    return file_name


def create_top_level_path(filename2append: str, *args: str) -> str:
    """ Create path starting in Top Level of SMIDA. """

    local_path = os.path.join(get_top_level_path(), *args)

    return create_n_append(local_path, filename2append)


def create_user_path(filename2append: str, *args: str) -> str:
    """ Create path starting in user Home folder. """

    local_path = os.path.join(Path.home(), 'SMIDAsettings', *args)

    return create_n_append(local_path, filename2append)


def create_n_append(local_path: str, filename2append: str) -> str:
    """ create path if it doesn't exist and append filename """

    if not os.path.isdir(local_path):
        os.makedirs(local_path)

    return os.path.join(local_path, filename2append)


def get_top_level_path() -> str:
    """
    :return path to top level folder of SMIDA:
        (without slash at the end)
    """
    # get path of this file (remove symbolic links)
    file_path = os.path.realpath(__file__)
    # step 3 folders up. This has to be changed, when this file is moved.
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    return file_path


def get_local_files(local_keyword: str) -> list:
    """ Scan some local folder and return list of local filenames

    :param local_keyword: string
        It can start with the keywords stored in the "local_keywords" variable in general/io_functions.py
    """

    assert local_keyword in local_keywords
    folderpath = replace_local_path(local_keyword)

    if not os.path.isdir(folderpath):
        return []

    filenames = os.listdir(folderpath)

    return [local_keyword + filename for filename in filenames if filename[0] != '.']


def remove_file(filepath: str) -> None:
    """ Remove possible local file. """

    filepath = replace_local_path(filepath)
    os.remove(filepath)

    return
