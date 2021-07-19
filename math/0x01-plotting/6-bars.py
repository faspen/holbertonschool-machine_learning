#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
pos = np.zeros(3)

for i in range(len(fruit)):
    plt.bar(
        np.arange(
            len(names)),
        fruit[i],
        0.5,
        bottom=pos,
        color=colors[i],
        label=fruits[i])
    pos += fruit[i]

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.yticks(np.arange(0, 90, step=10))
plt.xticks(np.arange(len(names)), names)
plt.legend(loc="upper right")
plt.show()
