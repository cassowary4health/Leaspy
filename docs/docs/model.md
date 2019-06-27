---
title: Documentation
summary: User documentation
---

# User documentation

## Model

Let's consider a model

$$y=f_{\theta}(z, x)$$

where

- $y$ corresponds to the output data
- $x$ corresponds to the input data
- $z$ corresponds to the hidden variables
- $f$ corresponds to the model
- $\theta$ corresponds to the model parameters


The model is able to do the following actions :

- `leaspy()` : defines $f$ with $\theta:= \theta_0$
- `leaspy.from_model_settings(model_settings)` : defines $f$ with $\theta:= \theta_{\text{model_settings}}$
- `leaspy.save(path)` : saves $\theta$ in the `path` directory
- `leaspy.fit(data, algo_settings)` : launches the model so that $\theta_{\text{init}} \xrightarrow{converge} \theta$ thanks to the data and settings
- `leaspy.predict(data, prediction_settings)` : returns the best hidden variable $z$ such that it minimizes $y^{\text{pred}} - f_{\theta}(x, z)$
- `leaspy.simulate(simulation_settings)` : generates collections of $(z, x)$ and therefore $(f_{\theta}(z, x))_{(z, x)}$
##
