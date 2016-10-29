import pandas as pd
import sys, os


"""
0  object internal database ID
2  object full name/designation
4  object IAU name
6  object NEO flag (Y/N)
7  potentially Hazardous asteroid
8  absolute magnitude parameter
9  magnitude slope parameter
15 object diameter
18 rotation period
19 GM product of mass and gravitational constatnt
32 eccentricity
33 semimajor axis
34 perihelion distance
35 inclination
36 longitude of the ascending node
37 argument of perihelion

44 Earth Minimum Orbit Intersection Distance
47 Jupiter Tisserand invariant

"""

# [0,4,6,7,8,9,15,18,19,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]

folder_path = sys.path[0]
database_path = os.path.join(folder_path, os.path.dirname(""),"./asteroid_data/latest_fulldb.csv")
database = pd.read_csv(database_path, sep=',', 
           usecols=range(28,35),low_memory=False)
# database = pd.read_csv(database_path, sep=',', usecols=['a', 'e', 'i', 'w', 'om', 'q',
#                                                         'H', 'neo', 'pha', 'moid'],
#                                                         low_memory=False)

db_head = database[:10]
print "db_head:\n", db_head