#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.suptitle('All in One')

# 0
plt.subplot(321)
plt.plot(y0, c='r')
plt.xlim(0, 10)
plt.ylim(0, 1000)
plt.xticks(np.arange(0, 12, step=2))

#1
plt.subplot(322)
plt.title("Men's Height vs Weight", size="x-small")
plt.xlabel("Height (in)", size="x-small")
plt.ylabel("Weight (lbs)", size="x-small")
plt.scatter(x1, y1, c='m', s=10)

# 2
plt.subplot(323)
plt.title("Exponential Decay of C-14", size="x-small")
plt.xlabel("Time (years)", size="x-small")
plt.ylabel("Fraction Remaining", size="x-small")
plt.yscale('log')
plt.xlim(0, 28800)
plt.plot(x2, y2)

# 3
plt.subplot(324)
plt.title('Exponential Decay of Radioactive Elements', size="x-small")
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.xlabel("Time (years)", size="x-small")
plt.ylabel("Fraction Remaining", size="x-small")
plt.yticks(np.arange(0.0, 1.5, step=0.5))
plt.plot(x3, y31, c='r', linestyle='dashed', label='C-14')
plt.plot(x3, y32, c='g', label='Ra226')
plt.legend()

# 4
plt.subplot(313)
plt.title('Project A', size="x-small")
plt.xticks(np.arange(0, 110, step=10))
plt.yticks(np.arange(0, 35, step=10))
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xlabel("Grades", size="x-small")
plt.ylabel("Number of Students", size="x-small")
plt.hist(student_grades, bins=10, edgecolor="black", range=(0, 100))

plt.show()
