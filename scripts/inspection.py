""" Try to catch some common erros """

import sys
import pkg_resources

# Verify Python Version #
#########################


##########################
# Working Python version #
##########################
#  add more, if it works for you
working_python_versions = ['3.6.9', '3.7.3', '3.7.4', '3.7.5', '3.7.7', '3.7.8']  # 3.7.8 is recommended.

##########################
# Crashing Python version #
##########################
#  add more, appends when you have problems
# noinspection PyListCreation
crashing_python_versions = []

# Python 3.8 shows issues with anaconda!

crashing_python_versions.append('3.5.2')
# Martin 28.06.19:
# python3 scripts/evaluation.py
#   File "scripts/evaluation.py", line 40
#     print(f'ERROR: unknown dataset: {dataset_name}')
#                                                   ^
# SyntaxError: invalid syntax


def check_python_version(verbose=False):
    # give warning if not a supported version
    python_version = "{}.{}.{}".format(*sys.version_info[0:3])

    if python_version in working_python_versions:
        if verbose:
            print("Your Python version ({}) is compatible!".format(python_version))
        return
    elif python_version in crashing_python_versions:
        print("WARNING: You are using Python {}. This is noted as crashing!".format(python_version))
    else:
        print("You are using Python {}.".format(python_version) +
              " If everything works, can you add it to 'working_python_versions' in scripts/inspection.py?")


# Verify Other Packages #
#########################

def get_version_number(package_name: str) -> str:
    """ Try to get version number or print warning """
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound():
        print("WARNING: {} was not found!")


def check_scikit():
    version_number = get_version_number('scikit-learn')
    if version_number != '0.22.2.post1':
        print('WARNING: scikit-learn is at "{}" instead of 0.22.2.post1.\n'
              '         This might be a problem for loading trained Classifiers!'.format(version_number))


if __name__ == "__main__":
    check_python_version(verbose=True)
    check_scikit()
