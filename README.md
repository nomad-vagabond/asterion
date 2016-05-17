# Asterion
A library for prediction of undiscovered potentially hazardous asteroids' orbital parameters using machine learning and computer simulations.

## CYA
Web application for analysis of asteroids on the subject of being potentially hazardous:
http://nasa2016kr.pythonanywhere.com/
Source available at: https://github.com/nasa2016kr/cya

## Asteroid Database
Library uses data from the database provided by Ian Webster. On the first use of the library the latest asteroid database will be downloaded automatically. If you are using proxy or something else prevents automatic download use this link: http://www.ianww.com/latest_fulldb.csv to get the latest database and put it into ./asteroid_data folder.

## How To Use

1. Clone this repository to your hard drive.

2. If you are using IPython or Jupyter Notebook open asterion.ipynb in notebook.

    Otherwise run scripts in next sequence:
    - read_database.py (Load asteroid database and calculate MOID)
    - generate_orbits.py (Generate virtual asteroid orbits and calculate MOID)
    - asterion_learn.py (Train classifier on virtual asteroids and test it on real data)

## Requirements

- Python >= 2.7, 
- NumPy >= 1.8.1
- SciPy >= 0.13.3
- Pandas >= 0.18.0
- scikit-learn >= 0.17.1
- Matplotlib >= 1.3.1
- IPython >= 3.0.0 (optional)
- PyOpenGL >= 3.0.2

*Tested on Ubuntu 14.04*

