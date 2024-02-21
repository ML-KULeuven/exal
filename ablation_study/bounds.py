
import math
import random
from program import Program
from sample import Query
import matplotlib.pyplot as plt

ITER = 40
SMOOTH = 500

p = Program("./programs/bounds/models1.txt")
q = p.atoms["q"]

def complete(program, values):
    for a in program.atoms.values():
        if a.fact:
            if not values.has(a):
                values.set(a, random.random() < 0.5)
        elif values.has(a):
            values.clear(a.name)
    return tuple(sorted(values.values.items()))

def probability(program, values):
    prob = 1
    for a, v in values:
        if v:
            prob *= program.atoms[a].definition
        else:
            prob *= 1-program.atoms[a].definition
    return prob

x = range(ITER)
def gety(factor=1, agree=True):
    yps = []
    yns = []
    for _ in range(SMOOTH):
        counts = {}
        pos = [complete(p, Query(factor=factor, counts=counts).execute(p.atoms["q"], True)) for _ in range(ITER)]
        counts = {}
        neg = [complete(p, Query(factor=factor, counts=counts).execute(p.atoms["nq"], True)) for _ in range(ITER)]
        yp = []
        yn = []
        for i in x:
            sofar = set(pos[:i+1])
            if agree:
                s = sum(probability(p, e) for e in sofar)
            else:
                s = math.exp(sum(math.log(len(sofar) * probability(p, e)) for e in sofar) / len(sofar))
            yp.append(s)
            sofar = set(neg[:i+1])
            if agree:
                s = sum(probability(p, e) for e in sofar)
            else:
                s = math.exp(sum(math.log(len(sofar) * probability(p, e)) for e in sofar) / len(sofar))
            yn.append(1-s)
        yps.append(yp)
        yns.append(yn)
    return [sum(y[i] for y in yps) / SMOOTH for i in range(ITER)], [sum(y[i] for y in yns) / SMOOTH for i in range(ITER)]

yp, yn = gety(1)
ypd, ynd = gety(0.2)
ypna, ynna = gety(1, agree=False)
ypda, ynda = gety(0.2, agree=False)

MS = 16
ME = 5

plt.plot(x, yp, color="r", label="Uniform / Agree", marker="o", markersize=MS, markevery=ME)
plt.plot(x, yn, color="r", marker="o", markersize=MS, markevery=ME)
plt.plot(x, ypna, color="r", linestyle="dashed", label="Uniform / No-agree", marker="d", markersize=MS, markevery=ME)
plt.plot(x, ynna, color="r", linestyle="dashed", marker="d", markersize=MS, markevery=ME)
plt.plot(x, ypd, color="b", label="Diverse / Agree", marker="o", markersize=MS, markevery=ME)
plt.plot(x, ynd, color="b", marker="o", markersize=MS, markevery=ME)
plt.plot(x, ypda, color="b", linestyle="dashed", label="Diverse / No-agree", marker="d", markersize=MS, markevery=ME)
plt.plot(x, ynda, color="b", linestyle="dashed", marker="d", markersize=MS, markevery=ME)
plt.plot([0, ITER-1], [max(yp), max(yp)], color="k", label="True objective")
plt.legend(fontsize=28)
plt.ylim((0, 1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Objective", fontsize=32)
plt.xlabel("Samples", fontsize=32)
plt.show()
