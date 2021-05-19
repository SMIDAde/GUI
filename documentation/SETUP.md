# SETUP: Schau mir in die Augen
[List of Documentation Files](menu.md)

[TOC]

## Preparation

You need:

-  **Python**
  It is suggested to have **Python3.7** installed.
  See versions listed in the `scripts/inspection.py` file

| Task | Ubuntu | macOS |
| --------- | --------- | --------- |
| install Git lfs | `sudo apt-get install git-lfs`<br />Some errors during git-lfs installation can be ignored if the installation completed successfully |  ? |

- Windows users: Check [Windows](Windows.md) for help. (e.g. run make commands)

### Creating Environment and Fulfilling Requirements

If you create an virtual environment with PyCharm, you can activate it from the terminal by:

``` bash
source venv/bib/activate
```

Every following command is run inside this.
*This might be similar with docker and other environments. Please improve this documentation if you can.*

On **Ubuntu** you can install packages by using `sudo pip3 install <name of package>` (or `pip`)

- `sudo pip3 install -r requirements.txt` should install all needed Packages

**Hint**: Make sure to check pythonpath is addressed correctly
`https://stackoverflow.com/questions/11960602/how-to-add-something-to-pythonpath`

Maybe it is helpful to run: `export PYTHONPATH=$PWD` when you are in the SMIDA Repository or set it manual.

### Using Markdown Documentation

This documentation you are reading can be used with [typora](https://typora.io/) for maximal comfort.

- [ TOC ] is translated to the directory of contents

## Using with PyCharm

### Visualization

You want to Run: **schau_mir_in_die_augen/visualization/gui_bokeh.py** 

In `Edit Configuration` add the following parameters:

```
Module name: bokeh
Parameters: serve gui_bokeh.py --dev
```
Make sure a compatible Python version (e.g. **Python 3.7**) is added and selected in `Edit Configuration`

### Known Errors

-  Ubuntu: SIGKILL
	Happens when memory is not enough and process gets killed.
-  Python Version is a common problem. Maybe you want to run:  
    	`python scripts/inspection.py`
        	and try different Versions.
