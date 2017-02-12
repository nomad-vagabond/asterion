# Asterion

Asterion is a project that aims at outlining subgroups of all major near-Earth asteroid groups with high concentration of potentially hazardous in them. The process of discovering boundaries of these subgroups is based on the application of machine learning (ML) techniques to mine asteroid database and  group together potentially hazardous asteroids (PHAs) by their orbital parameters. To obtain meaningful and easy-understandable results the approach presented here seeks after boundaries of the PHA subgroups in 2- and 3-dimensional subspaces of orbital parameters. It allows to outline ~90% of all existing and hypothetical PHAs into subgroups with high purity (~0.9). The ensemble of these subgroups provides a unique insight into the possible residences of yet undiscovered PHAs.

Asterion consists of:
- a set of Python modules for mining asteroid database and visualizing results;
- a set of Jupyter notebooks representing the database mining flow.


## Asteroid database
Asterion uses data from the asteroid database provided by Ian Webster (see https://github.com/typpo/asterank). On the first use the latest asteroid database will be downloaded automatically. If you are using proxy or something else prevents automatic download use this link: http://www.ianww.com/latest_fulldb.csv to get the latest database and put it into ./asteroid_data folder. Make sure its name is "latest_fulldb.csv".

## Asterion modules
Essentially a set of Python modules aimed at simplifying the process of the asteroid database mining. It includes:

- *asterion_learn* - collection of interfaces to handle ML-related operations
- *calculate_orbits* - collection of functions for calculating orbit intersection distances
- *generate_orbits* - API for generating virtual asteroids
- *learn_data* - collection of functions for preparation of training and testing datasets
- *read_database* - collection of functions for handling basic operations with the asteroid database
- *visualize_data* - collection of data visualization functions

## Database mining approach

The approach of extraction the PHA subgroups is represented in the next Jupyter notebooks:

1. *mine_neas* - load asteroid database, extract NEAs, generate virtual asteroids, split NEAs into 4 domains
2. *mine_domain1* - extract PHA subgroups in the 1st domain
3. *mine_domain2* - extract PHA subgroups in the 2nd domain
4. *mine_domain3* - extract PHA subgroups in the 3rd domain
5. *mine_domain4* - extract PHA subgroups in the 4th domain

Pay attention that some operations in the asteroid database mining chain are **time-consuming**. It means that their execution may take from several minutes up to **several hours**. The whole chain in total can take up to **8 hours of computations** depending your machine. You can decrease computation complexity by decreasing the number of virtual asteroids, but this will lead to the need of finding alternative values for the ML algorithms' input parameters in order to get desired results.

## Requirements

Python version 2.7 is required and next Python packages must be installed in your system to run Asterion:

- numpy >= 1.8.1
- scipy >= 0.14.1
- pandas >= 0.18.0
- scikit-learn >= 0.17.1
- matplotlib >= 1.3.1

The Jupyter Notebook (version 4.2.0 or higher) is also required.

*tested on Ubuntu 14.04 and Windows 7*

## History

The project was initiated at NASA Space Apps Challenge global hackathon in April 2016. It was a part of a double project named Asterion-CYA that became a global finalist in the nomination “Best use of data”. Second part of the project was a web application (https://github.com/nasa2016kr/cya) for checking the PHA state of asteroids and rendering their orbits. 

## License (MIT)

Copyright (c) 2016 Vadym Pasko (keenon3d@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.