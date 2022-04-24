import numpy as np

A = np.matrix([[1, 0, 1], [-2, 1, 0], [2, -1, 1]])
B = np.matrix([[2, 1, 0], [1, 1, 2], [2, 0, 2]])


print("A = \n{}".format(A))
print("B = \n{}".format(B))
print("A + B = \n{}".format(A + B))
print("A - B = \n{}".format(A - B))
print("A * B = \n{}".format(A * B))
print("A*-1 = \n{}".format(A**-1))