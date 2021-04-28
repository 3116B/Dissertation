import numpy as np

# Parameters
n_b = 1
n_g = 1
n = n_g + n_b
y = 0.4

# Quadratic parameters
a = 1
b = -(n*y+1-n_g/n_b)
c = -n_b*y

# Root properties
determinant = b**2 - 4*a*c

if determinant < 0:
    print(determinant)

try:
    numerator = -b + np.sqrt(determinant)
except:
    print(f'determinant is {determinant} which should not be negative')

denominator = 2*a

answer = np.divide(numerator, denominator)

print(answer)
