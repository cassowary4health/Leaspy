[![pipeline status](https://gitlab.com/icm-institute/aramislab/leaspy/badges/master/pipeline.svg)](https://gitlab.com/icm-institute/aramislab/leaspy/commits/master)
[![Documentation Status](https://readthedocs.org/projects/leaspy/badge/?version=latest)](https://leaspy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/leaspy.svg)](https://badge.fury.io/py/leaspy)

# Leaspy - LEArning Spatiotemporal Patterns in Python
Leaspy is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.

## Get started Leaspy
#### OS
- Mac and Linux - check for windows

#### Dependencies

- Python (>=3.6)
- torch (>=1.2.0, <1.7)
- numpy (>=1.16.6)
- pandas (>=1.0.5)
- scipy (>=1.5.4)
- scikit-learn (>=0.21.3, <0.24)
- joblib (>=0.13.2)
- matplotlib (>=3.0.3)
- statsmodels (>=0.12.1)

#### Installation

1. Create a dedicated environment (optional):

Using `conda`
```
conda create --name leaspy python=3.7
conda activate leaspy
```

Or using `pyenv`
```
pyenv virtualenv leaspy
pyenv local leaspy
```

2. Install leaspy
`pip install leaspy`



## Description
Leaspy is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.
Considering these series of short-term data, the software aims at :
- recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- positioning each patient observations relatively to the group-average timeline, in term of both temporal differences (time shift and acceleration factor) and spatial differences (diffent sequences of events, spatial pattern of progression, ...)
- quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- imputing missing values
- predicting future observations
- simulating virtual patients to unbias the initial cohort or mimis its characteristics



The software package can be used with scalar multivariate data whose progression can be modeled by a logistic shape, an exponential decay or a linear progression.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.


#### Further information
More detailed explanations about the models themselves and  about the estimation procedure can be found in the following articles :

- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations*. Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman.  The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access PDF](https: //hal.archives-ouvertes.fr/hal-01540367).
- **Application to imaging data**: *Statistical learning of spatiotemporal patterns from longitudinal manifold-valued networks*. I. Koval, J.-B. Schiratti, A. Routier, M. Bacci, O. Colliot, S. Allassonnière and S. Durrleman. MICCAI, September 2017. [Open Access PDF](https://arxiv.org/pdf/1709.08491.pdf)
- **Application to imaging data**: *Spatiotemporal Propagation of the Cortical Atrophy: Population and Individual Patterns*. Igor Koval, Jean-Baptiste Schiratti, Alexandre Routier, Michael Bacci, Olivier Colliot, Stéphanie Allassonnière, and Stanley Durrleman. Front Neurol. 2018 May 4;9:235. Open Access PDF
- **Intensive application for Alzheimer's Disease progression**: *AD Course Map charts Alzheimer's disease progression*, I. Koval, A. Bone, M. Louis, S. Bottani, A. Marcoux, J. Samper-Gonzalez, N. Burgos, B. Charlier, A. Bertrand, S. Epelbaum, O. Colliot, S. Allassonniere & S. Durrleman, Under review [Open Access PDF](https://hal.inria.fr/hal-01964821/document)
- www.digital-brain.org: website related to the application of the model for Alzheimer's disease

## Supported features
- `fit` : determine the **population parameters** that describe the disease progression at the population level
- `personalize` : determine the **individual parameters** that characterize the individual scenario of biomarker progression
- `estimate` : evaluate the biomarker values of a patient at any age, either for missing value imputation or future prediction
- `simulate` : generate synthetic data from the model


### Examples & Tutorials
The `example/start/` folder contains a starting point if you want to launch your first scipts and notebook with the Leaspy package.
You can find additional description in this [Medium post](https://medium.com/@igoroa/analysis-of-longitudinal-data-made-easy-with-leaspy-f8d529fcb5f8) (Warning: The plotter and the individual parameters described there have been deprecated since then)

### Documentation
https://leaspy.readthedocs.io/en/latest/

### Website
[Coming soon]

## Support

The development of this software has been supported by the European Union H2020 program (project EuroPOND, grant number 666992, project HBP SGA1 grant number 720270), by the European Research Council (to Stanley Durrleman project LEASP, grant number 678304) and by the ICM Big Brain Theory Program (project DYNAMO).

## Licence

The package is distributed under the GNU GENERAL PUBLIC LICENSE v3.

## Contacts
http://www.aramislab.fr/
