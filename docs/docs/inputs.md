---
title: Inputs
summary: Inputs of the model
---

# Data

The format of the input data is key to use Leaspy.

### Data csv

The user is required to provided a specific data structure to the data object.

| ID | TIME | name_of_feature_1 | ... | name_of_feature_m |
| ------------ | ------------- | ------------ | ------------ | ------------ |
| id_of_the_subject | age_of_the_subject | value_of_feature_1 | ... | value_of_feature_m |

An exemple is given below :

| ID | TIME | bio_1 | ... | bio_q |
| ------------ | ------------- | ------------ | ------------ | ------------ |
| id_1 | 75.1  | 1.2 | ... | 40.1 |
| id_1 | 77.3  | -0.8 | ... | 48.1 |
| id_1 | 79.7  | 1.5 | ... | 49. |
| id_2 | 60.  | .2 | ... | 25.7 |
| id_2 | 62.1  | 1.8 | ... | 29.5 |
| id_3 | 69.1  | 1.7 | ... | 15.3 |
| ...  | ... | ... | ... | ... |
| id_n | 80.1  | 2.4 | ... | 39.8 | 
| id_n | 83.5  | 2.3 | ... | 39.7 | 

### Data object

Lorem Ipsum

### Internal dataset structure

For optimization : n-dimensionnal array (numpy or pytorch)


# Model Settings

The model settings corresponds to all the variables needed by the model either to be instantiated or to be loaded from a previous state that has been saved in json.

### model_settings.json

A model can either be instantiated randomly given a set of variables, or from a `json` file that fully describes a model.
This `json` file has essentially three keys :

- the `type` of the model (univariate, multivariate, ...)
- the `parameters` of the model, i.e. a dictionary of the parameters $\theta$ that best describes the model
- the `hyperparameters` of the model that are essentially all the other variables needed to initialized the model

### model_settings object

The `model_settings` object is used to read the `json` file.


# Algorithm Settings

### algorithm_settings.json

Lorem Ipsum

### algorithm_settings object

Lorem Ipsum
