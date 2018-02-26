import random as ran
import math
import numpy as np

genSize = 100
inputSize = 1
nodes = [5, 5, 5, 5]  # [4,3]for two hidden levels with 4 and 3 nodes
outputs = 1
generations = 100


def sigmoid(x, a):
    return (a * x) / math.sqrt(1 + x ** 2)


def inputs():
    a = []
    for _ in range(inputSize):
        cm = ran.randrange(1, 4)
        a.append(cm)
    return a


def score(val, inp):
    o = 0
    if val > (1 / 3):
        o = val / abs(val)
    vs = inp[len(inp) - 1]
    if o == vs:
        return 0
    elif o == vs - 2 or (o == -1 and vs == 3):
        return 1
    else:
        return -1


def test(org, inp):
    ins = inp[:]
    for n in range(len(nodes) + 1):
        outs = []
        piv2 = 0
        if n == len(nodes):
            piv2 = outputs
        else:
            piv2 = nodes[n]
        for j in range(piv2):
            op = []
            p = 0
            if n == 0:
                p = len(ins)
            else:
                p = nodes[n - 1]
            for i in range(p):
                g = ins[i]
                t = (g * org[n][j][i])
                op.append(t)
            o = 0
            for g in op:
                o += g
            outs.append(sigmoid(o, 1))
        ins = outs[:]
    o = ins[0]
    return score(o, inp[:])


def rando():
    return ran.random() * 2 - 1


def create(inp):
    gener = []
    for _ in range(genSize):
        org = [0 for _ in range(len(nodes) + 1)]
        for k in range(len(nodes) + 1):
            if k == len(nodes):
                org[k] = [0 for _ in range(outputs)]
                for i in range(outputs):
                    if k == 0:
                        org[k][i] = [0 for _ in range(inputSize)]
                        for j in range(inputSize):
                            org[k][i][j] = rando()
                    else:
                        org[k][i] = [0 for _ in range(nodes[k - 1])]
                        for j in range(nodes[k - 1]):
                            org[k][i][j] = rando()
            else:
                org[k] = [0 for _ in range(nodes[k])]
                for i in range(nodes[k]):
                    if k == 0:
                        org[k][i] = [0 for _ in range(inputSize)]
                        for j in range(inputSize):
                            org[k][i][j] = rando()
                    else:
                        org[k][i] = [0 for _ in range(nodes[k - 1])]
                        for j in range(nodes[k - 1]):
                            org[k][i][j] = rando()
        gener.append(org[:])
    return gener


def refill(past):
    l = len(past)
    new = [[0 for y in range(len(past[0]))] for z in range(genSize - l)]
    for k in range(genSize - l):
        for i in range(len(past[0])):
            new[k][i] = [0 for x in range(len(past[0][i]))]
            for j in range(len(past[0][i])):
                new[k][i][j] = [0 for x in range(len(past[0][i][j]))]
                for m in range(len(past[0][i][j])):
                    mom = ran.randint(0, l - 1)
                    rand = np.random.normal()
                    if rand > 0:
                        radn = sigmoid(rand, 1 - past[mom][i][j][m])
                    else:
                        rand = sigmoid(rand, 1 + past[mom][i][j][m])
                    new[k][i][j][m] = past[mom][i][j][m] + rand
    return past + new


inp = inputs()
print("input:", inp)
gen = create(inp)


def tester(a, b, t):
    o = test(a, b)
    if o == 1:
        t[0] += 1
    return o


for i in range(generations):
    if len(gen) < genSize:
        gen = refill(gen[:])
    t = [0]
    gen.sort(key=lambda r: tester(r[:], inp[:], t), reverse=True)
    print(t[0])
    gen = gen[:math.floor(len(gen) / 2)]
    inp = inputs()
