[List of Documentation Files](menu.md)

# Data visualization with BOKEH

[TOC]

## General

### Installation

If not already installed, run `pip3 install bokeh`.  
Don't use `pip install bokeh ` for Python 2.7, you want a newer python version.  
You can check your python version used by bokeh with `bokeh info`.

### Start

To start a BOKEH Server write `make bokeh` in Terminal.  
Now you can use the returned link to acess it.  
To stop use "Strg + C" in Pycharm.

The accessed script is "schau_mir_in_die_augen/visualization/gui_bokeh.py"

## Description of Tool

See [Visualization Paper](https://gitlab.informatik.uni-bremen.de/cgvr/smida2/visualization_paper) and [Video](https://seafile.zfn.uni-bremen.de/f/52b979ec4f114519b6bd/).

## Interface

### Settings

| Section | Content |
| --- | --- |
| Headline | Color of Users |
| Save/Load | saving/loading of Settings |
| Select Data | |
| Select Annotation | Switch Trajectory, Saccades, Fixation and Counts on/off. Change visualization of Duration |
| Animation | Beside Animation: Reset Time, Mark Trajectory (ends) and Ping (find trajectory) |
| Coustomization | One Color for all Fixations Numbers and Markers, Coloring of Image. |

#### Load and Save Settings
To Load/Save Settings enter every name or path you want and press load/save.
With `[bet]xxx` you will use the general path in the repository. To commit your file for others, you have to manually add it in [Git](GITLAB.md).
With `[bus]xxx` you will use your home directory. Click `Open Folder` to get there.

If you want to load a specific view, you first have to move the images in plot to deactivate autoranging.
Than you can click "load view".

### Trajectory Plot (+ Feature Table)

Below the Trajectory Plot there are the settings to modify the data:
- interpolation
- Smoothing
- Removal of duplicates
- centering
- visual cropping

Bewlow the Feature Table there is:
- data-scaling
- x/y- offset

### Velocity Plot

Only Settings for IVT

### Prediction

| Columns | Content |
| --- | --- |
| 1 | selecting and loading the Evaluator |
| 2 | Showing the results, select the user and set the Parameter |
| 3 | information of classification |
| 4 | information of classification |
| 5 | Prediction Correctness Map |

#### Prediction Correctness Map

The binning of the PCVs into a Heatmap is not trivial.
Usually the bad values overweight the good and suppress information.

Therefore you can choose how you want to show it:

| Settings for PCM | Meaning for each bin                                         |
| ---------------- | ------------------------------------------------------------ |
| full             | Sum of all related PCV values                                |
| full-sum         | Sum of positive/negative PCV after each are normalized by sum of them |
| full-max         | ... normalized by max/min value                              |
| full-len         | ... normalized by number of values                           |
| good             | Map only beneficial PCV                                      |
| bad              | Map only harmful PCV (in red)                                |
| both             | Show red and green PCM                                       |



### Training

Select Parameters and directly train evaluators.

## Applications


### Visualization of Prediction

1. Train some classifier (See [Training](EVALUATION.md) *todo: update this link*)
	- Make shure you save modelfile in [model]-folder
2. Start Bokeh
3. Scroll down and select your saved Evaluator.
4. Click "Use Parameters of Classifier"
5. Selected some trained user.
6. Click "Show Evaluation Result"
	- use the "Color Scale" to better divide the values.

### Debugging

Maybe look also in the [Setup](SETUP.md)

#### Configuration

In PyCharm you have to set the Run/Debug Configurations for "gui_bokeh.py":

|  |  |
| --- | --- |
| Script path (possible) | ```/home/USERNAME/.local/bin/bokeh``` |
| Module name (better alternative?) | bokeh |
| Parameters  | ```serve gui_bokeh.py --dev``` |

The working directory have to stay in the visualization folder.

#### Other Errors

- `pip install --upgrade --force-reinstall pillow`
- `pip install matplotlib`

- If you have an error while running **gui_bokeh** regarding **PIL** and it's not solved by installing **Pillow** (`pip install --upgrade --force-reinstall pillow` could be helpful), 
  then make sure you have initialized the modules by `git install lfs` and pulled by  `git lfs pull`.