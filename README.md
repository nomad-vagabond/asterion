# Asterion

A library for prediction of undiscovered potentially hazardous asteroids' orbital parameters using machine learning and computer simulations.

Web application for analysis of asteroids on the subject of being potentially hazardous:
https://n2k-1-nasa2016kr.c9users.io/
Source available at: https://github.com/nasa2016kr/cya


Repository doesn't include asteroid database as well as trained machine learnig classifier and virtual asteroid orbits used for simulation.

To use library:

1. Clone this repository to your hard drive.

2. Download asteroid database http://www.ianww.com/latest_fulldb.csv and put it to ./asteroid_data

3. Generate virtual asteroid orbits by running script generate_orbits.py

    You can vsualize generated orbits by running visualize_data_gl.py

4. Run asterion_learn.py to train classifier on virtual asteroids and test it on real dataset.

To work with library you need next python packeges being installed on your system:

- numpy
- scipy
- pandas
- sklearn
- matplotlib

Tested on Ubuntu 14.04





