.. figure::  leaspy_logo.png
  :align:   center

Welcome to Leaspy's documentation!
**********************************

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   nutshell
   contribute

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api

.. toctree::
..   :maxdepth: 2
..   :hidden:
..   :caption: Tutorial - Examples
..
..   auto_examples/index

.. toctree::
..   :maxdepth: 1
..   :hidden:
..   :caption: Additional Information
..
..   reproducibility
..   changelog

LEArning Spatiotemporal Patterns in Python
==========================================


Description
-----------
**Leaspy** is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.

.. figure:: leaspy_front.png
  :align:   center

Considering these series of short-term data, the software aims at :

- Recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- Positioning each patient observations relatively to the group-average timeline, in term of both temporal differences (time shift and acceleration factor) and spatial differences (diffent sequences of events, spatial pattern of progression, ...)
- Quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- Imputing missing values
- Predicting future observations
- Simulating virtual patients to unbias the initial cohort or mimic its characteristics

The software package can be used with scalar multivariate data whose progression can be modeled by a logistic shape, an exponential decay or a linear progression.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.

`Getting started <install.html>`_
---------------------------------

Information to install, test, and contribute to the package.

`User Guide <user_guide.html>`_
-------------------------------

The main documentation. This contains an in-depth description of all
algorithms and how to apply them.

`API Documentation <api.html>`_
-------------------------------

The exact API of all functions and classes, as given in the
docstrings. The API documents expected types and allowed features for
all functions, and all parameters available for the algorithms.

Further information
-------------------
More detailed explanations about the models themselves and  about the estimation procedure can be found in the following articles :


- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations.* Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman.  The Journal of Machine Learning Research, 18:1–33, December 2017. `Open Access <https: //hal.archives-ouvertes.fr/hal-01540367>`_
- **Application to imaging data**: *Statistical learning of spatiotemporal patterns from longitudinal manifold-valued networks.* I. Koval, J.-B. Schiratti, A. Routier, M. Bacci, O. Colliot, S. Allassonnière and S. Durrleman. MICCAI, September 2017. `Open Access  <https://arxiv.org/pdf/1709.08491.pdf>`_
- **Application to imaging data**: *Spatiotemporal Propagation of the Cortical Atrophy: Population and Individual Patterns.* Igor Koval, Jean-Baptiste Schiratti, Alexandre Routier, Michael Bacci, Olivier Colliot, Stéphanie Allassonnière, and Stanley Durrleman. Front Neurol. 2018 May 4;9:235. `Open Access <https://hal.inria.fr/hal-01910400>`_
- **Intensive application for Alzheimer's Disease progression**: *Simulating Alzheimer’s disease progression with personalised digital brain models*, I. Koval, A. Bone, M. Louis, S. Bottani, A. Marcoux, J. Samper-Gonzalez, N. Burgos, B. Charlier, A. Bertrand, S. Epelbaum, O. Colliot, S. Allassonniere & S. Durrleman, Under review `Open Access <https://hal.inria.fr/hal-01964821>`_
- `www.digital-brain.org <https://project.inria.fr/digitalbrain/>`_ : Website related to the application of the model for Alzheimer's disease.




.. `Examples <auto_examples/index.html>`_
.. --------------------------------------
..
.. A set of examples illustrating the use of the different algorithms. It
.. complements the `User Guide <user_guide.html>`_.
..
.. `Changelog <changelog.html>`_
.. ------------------------------
..
.. History of notable changes to the pyts.
..
.. See the `README <https://github.com/johannfaouzi/pyts/blob/master/README.md>`_
.. for more information.
..

Indices and tables
------------------

* :ref:`license`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`