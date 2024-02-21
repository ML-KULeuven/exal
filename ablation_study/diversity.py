
import pickle

import numpy as np
import torch

from matplotlib import pyplot as plt
from program import Program, Atom
from sample import Query, Request

class LearnQuery(Query):

    def __init__(self,
                 complete=True,
                 project=True,
                 factor: float = 1,
                 counts: dict[str, list[int]] = None):
        super().__init__(complete=complete, project=project, factor=factor, counts=counts)
        self.choices = []

    def _request(self, atom: Atom, value: bool) -> Request:
        return LearnRequest(atom, value)

class LearnRequest(Request):

    def choice(self) -> tuple[Atom, bool] | None:
        weights = np.power(self.query.factor, self.weights)
        if not np.any(weights):
            return None
        choice = np.random.choice(np.arange(len(self.atom.definition)), p=weights / np.sum(weights))

        self.query.choices.append((self.weights.copy(), choice, self.atom))
        self.query.counts[self.atom.name][choice] += 1
        self.weights[choice] = np.infty
        return self.atom.definition[choice]

def calculate_all(path, setups, factors, trials):
    results = {}
    for setup in setups:
        program = Program()
        program.read(f"./programs/diversity/{setup[0]}.txt")
        for factor in factors:
            print(f"calculating {setup[0]}-factor-{factor}")
            results[f"{setup[0]}-factor-{factor}"] = \
                calculate_setup(program=program, samples=setup[1], trials=trials, factor=factor)
    with open(path, "wb") as file:
        pickle.dump(results, file)

def calculate_setup(program, samples, trials, factor):
    atom = program.atoms["q"]
    diversity = [[] for _ in range(samples)]
    for _ in range(trials):
        counts = {}
        bag = set()
        for index in range(samples):
            query = Query(factor=factor, counts=counts)
            query.execute(atom, True)
            values = str(sorted((atom, value) for atom, value in query.values.items() if program.atoms[atom].fact))
            bag.add(values)
            diversity[index].append(len(bag))
    return [sum(point) / len(point) for point in diversity]

def learn_factor(program, trajectories, samples, iterations, eps):
    atom = program.atoms["q"]
    data = []
    for _ in range(trajectories):
        bag = set()
        choices = []
        counts = {}
        for _ in range(samples):
            query = Query(factor=1, counts=counts)
            query.choices = choices
            query.execute(atom, True)
            values = str(sorted((atom, value) for atom, value in query.values.items() if program.atoms[atom].fact))
            bag.add(values)
        data.append((choices, len(bag)))

    theta = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    optim = torch.optim.Adam([theta], lr=0.01)

    for _ in range(iterations):
        loss = 0
        factor = torch.sigmoid(theta)
        weight = torch.log(eps + factor)
        for point in data:
            for index in range(len(point[0]) - 2):
                inflow = point[0][index]
                outflow = point[0][index + 1]
                # factor ** inflow[0][inflow[1]]
                loss += weight * (torch.log(eps + sum(factor ** n for n in inflow[0])) - torch.log(eps + sum(factor ** n for n in outflow[0]))) ** 2
            inflow = point[0][-1]
            loss += weight * (factor ** inflow[0][inflow[1]] - point[1]) ** 2

        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        print(loss)
        print(factor)
    return torch.sigmoid(theta)

def visualise_all(path, setups, factors):
    with open(path, "rb") as file:
        results = pickle.load(file)
    m = {
        "branch-1": (0, "Branch (small)"),
        "branch-3": (1, "Branch (large)"),
        "bottom-1": (2, "Bottom up"),
        "many-1": (3, "Split"),
    }

    fig, ax = plt.subplots(1, 4, sharey=True)
    fig.set_figwidth(20)
    fig.supxlabel("Calls", fontsize=20, y=-0.005)

    for setup in setups:
        if setup[0] not in m:
            continue

        # plt.clf()
        # plt.cla()

        a = ax[m[setup[0]][0]]

        k = 0
        for factor in factors:
            k = max(k, results[f"{setup[0]}-factor-{factor}"][-1])
        for factor in factors:
            a.plot(range(setup[1]), [y / k for y in results[f"{setup[0]}-factor-{factor}"]], label=f"factor:{factor}")
        a.plot(range(setup[1]), [y / k for y in results[f"{setup[0]}-learn"]], label=f"learned", linestyle="dashed")
        x = range(setup[1])
        y = [1 - (1 - 1 / k) ** (n+1) for n in x]

        a.plot(x, y, label="uniform", color="k", linestyle="dashed")
        a.plot([0, 0, k-1, setup[1]], [0, 1/k, 1, 1], label="optimal", color="k", linewidth=1)
        a.tick_params(axis="x", labelsize=12)
        a.set_ylim([0.45, 1.05])
        # plt.yticks(fontsize=14)
        if a == ax[0]:
            a.set_ylabel("Diversity", fontsize=20)
        if a == ax[-1]:
            a.legend(fontsize=14)
        # plt.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.12)

        title = setup[0]
        if title in m:
            title = m[title][1]

        a.set_title(title, fontsize=20)
    plt.savefig(f"./output/diversity/plot-all.png")
    # plt.show()

TRIALS = 200

s = [
    ("branch-1", 50),
    ("branch-2", 100),
    ("branch-3", 400),
    ("many-1", 200),
    ("many-2", 1000),
    ("many-3", 5000),
    ("bottom-1", 60),
    ("bottom-2", 30),
    ("bottom-3", 200),
]
f = [1, 0.7, 0.4, 0.1]

calculate_all(path="./output/diversity/data.pickle", setups=s, factors=f, trials=TRIALS)
visualise_all(path="./output/diversity/data.pickle", setups=s, factors=f)
