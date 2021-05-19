# Evaluation of SMIDA
[List of Documentation Files](menu.md)

See [script_helper:call_train/call_test](../schau_mir_in_die_augen/evaluation/script_helper.py)
See [TestEvaluation](../schau_mir_in_die_augen/tests/test_evaluation.py)

---

This document gives an overview of evaluation options and the program structure of SMIDA.
There are other documents, if you are looking for an [OVERVIEW](OVERVIEW.md) of files or want to [SETUP](SETUP.md) the program. Further more Details are in [README](/README.md).

### Steps

Two methods are available: tain.py and evaluation.py

1.  Preparation (only settings)
	1.  Initialize [DATASETS](DATASETS.md)
	2.  Initialize [CLASSIFIER](CLASSIFIER.md): RandomForestClassifier or Rbfn
	3.  Initialize [METHOD](METHODS.md) with Classifier
2.  load_trajectories (slow)
3.  provide_feature
 4.  trajectory_split_and_feature (very slow, will calculate all feature)
5.  Either fit classifier (very slow) or predict_trajectory
6.  saving Results

### Parameters
You can run with these parameters:

| Parameter | Options or Default | Purpose |
| ------------- | ---------- | ---------- |
| method | 'paper-append'                                 | ... |
| dataset | 'bio-tex', 'bio-tex1y', 'bio-ran', 'bio-ran1y' | Select the data to train on. |
| clf | 'rf' , 'rbfn' | Select the [Classifier](Classifier) |
| *ul* | default=None | Limit of Users (randomly selected) |
| *seed* | default=42 | make random decisions repeatable |



All parameters written in *italic* style are optional.

Example for a valid call:
`evaluation.py --method our-one-clf --dataset bio-tex --classifier rf `