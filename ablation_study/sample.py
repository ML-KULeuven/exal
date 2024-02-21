
from __future__ import annotations

import numpy as np
from program import Atom, Assignment

class Query:
    """
    Class for executing a query in a logic program.

    TODO: param doc
    """

    def __init__(self,
                 complete=True,
                 project=True,
                 factor: float = 1,
                 counts: dict[str, list[int]] = None):
        self.complete = complete
        self.project = project
        self.factor = factor
        self.counts = {} if counts is None else counts

        self.values = None
        self.execution = None
        self.insertion = None

    def execute(self, atom: Atom, value: bool = True) -> dict[str, bool]:
        self.values = Assignment(atom.program, {atom.name: value})
        self.execution = self._request(atom, value)
        self.insertion = self.execution
        forward = True
        while self.execution is not None:
            forward = self.execution.forward() if forward else self.execution.backward()
        return self.values

    def request(self, atom: Atom, value: bool, requests: list["Request"]) -> bool:
        """
        Adds to the queue a request for the given atom to have the given value.
        As a side effect, the added request will also be put in the requests list.

        :param atom: The atom whose value is set.
        :type atom: Atom
        :param value: The desired value for the atom.
        :type value: bool
        :param requests: The list in which the added request will be put.
        :type requests: list[Request]
        :returns: Whether this assignment is conflict-free.
        :rtype: bool
        """

        if self.values.has(atom):
            return self.values.get(atom) == value
        self.values.set(atom, value)

        request = self._request(atom, value)
        requests.append(request)
        if not atom.fact:
            request.insert()
        return True

    def _request(self, atom: Atom, value: bool) -> Request:
        return Request(self, atom, value)

class Request:

    def __init__(self, query: Query, atom: Atom, value: bool):
        self.query = query
        self.atom = atom
        self.value = value

        self.next = None
        self.back = None
        self.weights = None
        self.requests = []

    def forward(self) -> bool:
        if self.query.insertion.next == self:
            self.query.insertion = self

        if self.value:
            for body in self.atom.definition:
                if not self.query.request(body[0], body[1], self.requests):
                    return False
        else:
            if self.atom.name not in self.query.counts:
                self.query.counts[self.atom.name] = np.zeros(len(self.atom.definition), dtype=np.float32)
            counts = self.query.counts[self.atom.name]
            self.weights = counts - np.min(counts)
            for index, body in enumerate(self.atom.definition):
                if self.query.values.has(body[0]):
                    if self.query.values.get(body[0]) == body[1]:
                        self.weights[index] = -1
                    else:
                        self.query.execution = self.next
                        return True
            body = self.choice()
            if not self.query.request(body[0], not body[1], self.requests):
                return False
        self.query.execution = self.next
        return True

    def backward(self) -> bool:
        for request in self.requests:
            self.query.values.clear(request.atom)
            request.delete()
        self.requests.clear()

        if not self.value:
            body = self.choice()
            if body is not None:
                if self.query.request(body[0], not body[1], self.requests):
                    self.query.execution = self.next
                    return True
                else:
                    return False
        self.query.execution = self.back
        return False

    def choice(self) -> tuple[Atom, bool] | None:
        weights = np.power(self.query.factor, self.weights)
        if not np.any(weights):
            return None
        choice = np.random.choice(np.arange(len(self.atom.definition)), p=weights/np.sum(weights))

        self.query.counts[self.atom.name][choice] += 1
        self.weights[choice] = np.infty
        return self.atom.definition[choice]

    def insert(self):
        self.next = self.query.insertion.next
        self.back = self.query.insertion
        self.query.insertion.next = self
        if self.next is not None:
            self.next.back = self
        if self.value:
            self.query.insertion = self

    def delete(self):
        if self.back is not None:
            self.back.next = self.next
        if self.next is not None:
            self.next.back = self.back
        if self.query.insertion == self:
            self.query.insertion = self.back
