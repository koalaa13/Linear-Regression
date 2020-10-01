import numpy
import matplotlib.pyplot
import random
import math


def scalar(a, b):
    res = 0.0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def sgn(u):
    return -1 if u <= 0 else 1


EPS = 1e-7


def calc_grad(w, ps):
    global m
    result = [0.0] * m
    for i in range(m):
        sum = 0.0
        for p in ps:
            g = scalar(w, p[0])
            y = p[1]
            den = abs(g) + abs(y)
            if den < EPS:
                continue
            c = (sgn(g - y) * den - sgn(g) * abs(g - y)) / (den ** 2)
            sum += 2 * p[0][i] * c
        result[i] = sum / len(ps)
    return result


def calc_smape(a, b):
    return 2 * abs(a - b) / (abs(a) + abs(b))


def calc_whole_smape(w, ps):
    sum = 0.0
    for p in ps:
        my_y = scalar(w, p[0])
        sum += calc_smape(my_y, p[1])
    return sum / len(ps)


def get_rand_arr(sz):
    result = []
    for i in range(sz):
        result.append(random.uniform(-1, 1))
    return result


file = open("3.txt", "r")

m = int(file.readline())
m += 1
n = int(file.readline())
training_dataset = []
f = []
ys = []

for i in range(n):
    row = [float(x) for x in file.readline().split()]
    f.append(row[:-1] + [1])
    ys.append(row[-1])
    training_dataset.append([row[:-1] + [1], row[-1]])

dataset = []
n1 = int(file.readline())
for i in range(n1):
    row = [int(x) for x in file.readline().split()]
    dataset.append([row[:-1] + [1], row[-1]])

step = 1e-3
w = [0.0] * m
graphic_step = 1000
next_w = [0.0] * m
max_iteration = 100000
for i in range(max_iteration):
    grad = calc_grad(w, training_dataset)
    for j in range(m):
        next_w[j] = w[j] - step * grad[j]
    w = next_w.copy()
    if i % graphic_step == 0:
        matplotlib.pyplot.plot(float(i), calc_whole_smape(w, training_dataset), "go")
        matplotlib.pyplot.plot(float(i), calc_whole_smape(w, dataset), "bo")
matplotlib.pyplot.savefig("gradient_descent.png")
matplotlib.pyplot.clf()

print("Gradient info:")
print("w:")
print(w)
print("SMAPE:")
print(calc_whole_smape(w, training_dataset))
print(calc_whole_smape(w, dataset))
print("------------------------")

u, s, vt = numpy.linalg.svd(f, full_matrices=False)
u = numpy.matrix.getT(u)
s = numpy.diag(s)
for i in range(len(s)):
    s[i][i] += 0.01
s = numpy.linalg.inv(s)
v = numpy.matrix.getT(vt)
fres = numpy.matmul(v, numpy.matmul(s, u))
q = numpy.matmul(fres, ys)
print("LMS info:")
print("q:")
print(q)
print("SMAPE:")
print(calc_whole_smape(q, training_dataset))
print(calc_whole_smape(q, dataset))
print("------------------------")

random.seed(13387)
rand_w = [0.0] * m
smape = calc_whole_smape(rand_w, training_dataset)
for i in range(max_iteration):
    grad = get_rand_arr(m)
    for j in range(m):
        next_w[j] = rand_w[j] - step * grad[j]
    cur_smape = calc_whole_smape(next_w, training_dataset)
    if cur_smape < smape:
        smape = cur_smape
        rand_w = next_w.copy()
    if i % graphic_step == 0:
        matplotlib.pyplot.plot(float(i), smape, "go")
        matplotlib.pyplot.plot(float(i), calc_whole_smape(rand_w, dataset), "bo")
matplotlib.pyplot.savefig("genetic_algo.png")
matplotlib.pyplot.clf()
print("genetic algo info:")
print("rand_w:")
print(rand_w)
print("SMAPE:")
print(calc_whole_smape(rand_w, training_dataset))
print(calc_whole_smape(rand_w, dataset))
print("------------------------")
