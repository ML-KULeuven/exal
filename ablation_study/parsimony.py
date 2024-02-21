
from program import Program, Atom
from sample import Query, Request

import numpy as np

class ConflictQuery(Query):

    def __init__(self,
                 complete=True,
                 project=True,
                 factor: float = 1,
                 counts: dict[str, list[int]] = None,
                 parsimony=True):
        super().__init__(complete=complete, project=project, factor=factor, counts=counts)
        self.parsimony = parsimony
        self.conflicts = 0

    def request(self, atom: Atom, value: bool, requests: list["Request"]) -> bool:
        if self.values.get(atom) != value:
            self.conflicts += 1
        return super().request(atom, value, requests)

    def _request(self, atom: Atom, value: bool) -> Request:
        return Request(self, atom, value) if self.parsimony else NoParsimonyRequest(self, atom, value)

class NoParsimonyRequest(Request):

    def __init__(self, query: Query, atom: Atom, value: bool):
        super().__init__(query=query, atom=atom, value=value)
        self.tries = 0

    def forward(self) -> bool:
        if self.query.insertion.next == self:
            self.query.insertion = self

        if self.value:
            for body in self.atom.definition:
                if not self.query.request(body[0], body[1], self.requests):
                    return False
        else:
            self.tries = 2 ** len(self.atom.definition)
            values = np.random.rand(len(self.atom.definition)) < 0.5
            i = np.random.randint(0, len(values))
            values[i] = not self.atom.definition[i][1]
            for body, value in zip(self.atom.definition, values):
                if not self.query.request(body[0], value, self.requests):
                    return False
        self.query.execution = self.next
        return True

    def backward(self) -> bool:
        for request in self.requests:
            self.query.values.clear(request.atom)
            request.delete()
        self.requests.clear()

        if not self.value:
            self.tries -= 1
            if self.tries > 0:
                values = np.random.rand(len(self.atom.definition)) < 0.5
                i = np.random.randint(0, len(values))
                values[i] = not self.atom.definition[i][1]
                for body, value in zip(self.atom.definition, values):
                    if not self.query.request(body[0], value, self.requests):
                        return False
                self.query.execution = self.next
                return True
        self.query.execution = self.back
        return False

TRIALS = 10000

p = Program("./programs/conflicts/top-0.txt")
a = p.atoms["q"]
c = []

for _ in range(TRIALS):
    q = ConflictQuery(parsimony=True)
    q.execute(a, True)
    c.append(q.conflicts)

print(f"{np.mean(c)} +- {np.std(c)}")
