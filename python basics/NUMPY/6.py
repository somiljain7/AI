
#/* NORMALIZING 5x5 MATRIX*/
import numpy as np
Z=np.random.random((5,5))
Zmax,Zmin = Z.max(),Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

