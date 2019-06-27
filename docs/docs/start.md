---
title: Documentation
summary: User documentation
---

## Get a new dataset and run the Model

```python
import leaspy as lp

data = read_data(data_path)
algo_settings = read_algo_settings(algo_settings_path)

model = lp(model_name)
model.fit(data, algo_settings)
model.save(path)
```

## Predict the individual parameters


```python
import leaspy as lp

data = read_data(data_path)
prediction_settings = read_prediction_settings(prediction_settings_path)

model = lp.from_model_settings(path_to_parameters)
individual_parameters = model.predict(data, prediction_settings)

# You can then perform your analysis on the individual parameters
```

## Simulate new parameters

```python
import leaspy as lp

simulation_settings = read_prediction_settings(simulation_settings_path)

model = lp.from_model_settings(path_to_parameters)
new_individuals = model.simulate(simulation_settings)
```
