import numpy as np

m = 100.4
h = 0.09
r = 0.175

ixx = m/12. * (3 * r**2 + h**2)
iyy = m/12. * (3 * r**2 + h**2)
izz = 0.5 * m * r**2

print ixx, iyy, izz