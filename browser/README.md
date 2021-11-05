# Description

This folder is intended to grasp some intuitions on modelling longitudinal data with Leaspy thanks to a dynamic webapp.

# How-to

0. Activate your leaspy virtual environment (conda/pyenv),
   e.g.: `conda activate leaspy` (cf. leaspy README.md)

1. Install browser specific requirements with:
   `pip install -r requirements.txt`

2. Launch the webapp server with:
   `python app.py &`

3. Open your browser and copy-paste the URL of the server

4. Play with the webapp:
   - load a model (examples are available in `browser/data/example`),
   - tweak individual par`meters manually,
   - or personalize the model to individual data that you add.

5. Kill the webapp server by using `kill %N` where N is id of job (cf. `jobs`)
