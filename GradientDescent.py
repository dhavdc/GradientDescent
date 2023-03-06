import numpy as np
from matplotlib import pyplot as plt

#Start - start value
#gradient_func - The gradient function (derivative)
#learning_rate - How much we scale by or the r
#max_iterations - A limit on how many jumps we want
#tolerance - Value to stop the algo at (how close to minima)

def gradient_descent(start, gradient_func, learning_rate, max_iterations, tolerance=0.01):
    steps = np.array([start])
    x = start

    for _ in range(max_iterations):
        diff = learning_rate * gradient_func(x)
        if np.abs(diff) < tolerance:
            break
        x = x - diff
        steps = np.append(steps, x)

    return steps, x

#Example 1 - Quadratic function
#f(x)=x^2+4x+1
#derivative - 2x + 4
def example_function_1(x):
    return x**2+4*x+1

def example_gradient_1(x):
    return 2*x+4

#Example 1 - Quasi-Convex
#f(x)=x^4-2x^3+2
#derivative - 4x^3-6x^2
def example_quasi_function_1(x):
    return x**4 - 2 * (x**3) + 2

def example_quasi_gradient_1(x):
    return 4 * (x**3)- 6 * (x**2)

history, result = gradient_descent(-1, example_quasi_gradient_1, 0.01, 100)

function_data_points = np.linspace(-1.5, 2.5, 100)
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(function_data_points, example_quasi_function_1(function_data_points))
plt.plot(history, example_quasi_function_1(history), marker="o")
for i in range(len(history)):
    ax.text(history[i] - 0.1, example_quasi_function_1(history)[i] - 1, i if i < 8 else '')
plt.show()


print(history)
print(result)


