# Baseline-Integrated normative modelling

This repository stores code and analyses for the preprint/published article entitled

> Predicting Developmental Norms from Baseline Cortical Thickness in Longitudinal Studies

Before running the code make sure to create a working python environment.

You can use our predefined `requirements.txt` file for this:

> conda create --name <your_env_name> python=3.12.9 --file requirements.txt

**Note:** we ran these analyses using a MacOS computer. Consider this using the `requirements.txt` file. 

## Order of Operations

**Note:** we cannot share the ABCD used on this repository.

1. Run the `preprocessing.ipynb` notebook.
2. Run the `modelling.ipynb` notebook.
3. Run the `plot_forward.ipynb` notebook.
4. Run the `stats.ipynb` notebook.
5. Run the `plot_stats_on_surface.ipynb` notebook.

Some functionality is provided within respective notebooks, others in `utils.py`.
Definition of normative models and class-functions are in `models.py`. 

The shell-scripts `concat*.sh` are for automate arranging figures produced by `plot_stats_on_surface`.
`plotter.py` provides some functionality for `plot_foward`.
