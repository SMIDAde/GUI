# Methods of SMIDA for Evaluation
[List of Documentation Files](menu.md)

```mermaid
graph TD
A(BaseEvaluation: base_evaluation.py) .-> B(EvaluationGeneralFixSac: evaluation_general.py)
B .-> Ab(OurEvaluationAppended: evaluation_our_appended.py)

```

[TOC]

## Overview

They are selected in *evaluation.py* by `-- method`

-  `paper-append`:  52 Feature features from "DBLP:journals/prl/GeorgeR16"

## Parent: BaseEvaluation

Parent for all evaluation classes

Some Important functions are brought by childs:

-  *trajectory_split_and_feature*
	Returns feature list.
-  *train*
-  *evaluation*
	Calls [*weighted_evaluation*](#weighted_evaluation) of BaseEvaluation (.5/.5 for Sac and Fix or 1 whren there is only one [CLASSIFIER](CLASSIFIER.md)).

### **load_trajectories**

Uses *dataset.load_training* and *dataset.load_testing* to load X,Y for training and testing data.
The Form is: X=Sample Arrays, Y=UserID with len(X)=len(Y).

With optional Parameter *limit* it can reduce number of datasets

### provide_feature

Gets a list of trajectories and labels

Will use *trajectory_split_and_feature* introduced by childs (see below) to get a list of values for multiple features of saccades and fixations.

This will take a while.
It is using the memory (cache) function of python to speed up future calls.



## Childs

The Seperation Algortihms are described in [SEPERATION](SEPERATION.md).

### train

Gets stuff from *provide_feature* and uses *clf.fit* from [CLASSIFIER](CLASSIFIER.md).



