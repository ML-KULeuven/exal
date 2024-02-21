
from __future__ import annotations

import re
import random

SYNTAX_BLANK = r"^\s*$"
SYNTAX_ATOM = r"^[a-z]\w*$"
SYNTAX_DEFINE = r"\s*<<\s*"
SYNTAX_BODY = r"\s*,\s*"
SYNTAX_NEGATION = "-"
SYNTAX_FACT = "f"

class Atom:
    """
    An atom in a logic program.
    TODO: params
    """

    def __init__(self, program: Program, name: str, definition: float | tuple[tuple["Atom", bool], ...]):
        self.program = program
        self.name = name
        self.fact = isinstance(definition, float)
        self.definition = definition

    def __str__(self):
        return self.name

class Program:
    """
    A logic program.
    TODO: params
    """

    def __init__(self, path: str = None):
        self.atoms = {}
        if path:
            self.read(path)

    def read(self, path: str) -> int:
        """
        Reads in a program from a file.

        :param path: The path of the file to read in.
        :type path: str
        :returns: The line at which an error occurred, or 0 if the file was read successfully.
        :rtype: int
        """

        with open(path, "r") as file:
            for index, line in enumerate(file, 1):
                if re.match(SYNTAX_BLANK, line):
                    continue
                line = re.split(SYNTAX_DEFINE, line.strip())
                if len(line) != 2:
                    return index

                name = line[0]
                if name in self.atoms or not re.match(SYNTAX_ATOM, name):
                    return index

                definition = line[1]
                if definition.endswith(SYNTAX_FACT):
                    try:
                        definition = float(definition[:-len(SYNTAX_FACT)])
                    except ValueError:
                        return index
                else:
                    bodies = []
                    for body in re.split(SYNTAX_BODY, definition):
                        negation = body.startswith(SYNTAX_NEGATION)
                        if negation:
                            body = body[len(SYNTAX_NEGATION):]
                        if body not in self.atoms:
                            return index
                        bodies.append((self.atoms[body], not negation))
                    definition = tuple(bodies)
                self.atoms[name] = Atom(self, name, definition)
        return 0

    def to_dimacs(self, path: str = None) -> str:
        """
        Converts the program to a CNF in DIMACS format.
        """

        lines = []
        variables = {name: str(index) for index, name in enumerate(self.atoms.keys(), 1)}
        for atom in self.atoms.values():
            if atom.fact:
                continue
            clause = f"{variables[atom.name]}"
            for body in atom.definition:
                if body[1]:
                    lines.append(f"-{variables[atom.name]} {variables[body[0].name]} 0")
                    clause += f" -{variables[body[0].name]}"
                else:
                    lines.append(f"-{variables[atom.name]} -{variables[body[0].name]} 0")
                    clause += f" {variables[body[0].name]}"
            lines.append(f"{clause} 0")
        program = f"p cnf {len(variables)} {len(lines)}\n" + "\n".join(lines)

        if path:
            with open(path, "w") as file:
                file.write(program)
        return program

    def to_problog(self, query: str = None, path: str = None) -> str:
        """
        Converts the program to a ProbLog program.
        """

        lines = []
        for atom in self.atoms.values():
            if atom.fact:
                lines.append(f"{atom.definition} :: {atom.name}.")
            else:
                body = ", ".join(f"{literal}" if value else f"\\+{literal}" for literal, value in atom.definition)
                lines.append(f"{atom.name} :- {body}.")
        if query:
            lines.append(f"query({query}).")
        program = "\n".join(lines)

        if path:
            with open(path, "w") as file:
                file.write(program)
        return program

class Assignment:
    """
    A (partial) assignment in a logic program.
    TODO: params
    """

    def __init__(self, program: Program, values=None):
        self.program = program
        self.values = {} if values is None else values

    def set(self, atom: Atom, value: bool):
        self.values[atom.name] = value

    def clear(self, atom: Atom):
        self.values.pop(atom.name)

    def has(self, atom: Atom):
        return atom.name in self.values

    def get(self, atom: Atom) -> bool:
        return self.values[atom.name]

    def project(self) -> "Assignment":
        self.values = {atom.name: self.values[atom.name] for atom in self.program.atoms.values() if atom.fact}
        return self

    def complete(self, uniform=True, project=True) -> "Assignment":
        for name, atom in self.program.items():
            if name in self.values or not atom.fact:
                continue
            self.values[name] = random.random() < (0.5 if uniform else atom.definition)
        if not project:
            self.entail()
        return self

    def entail(self):
        return self # TODO

if __name__ == "__main__":

    FILE = './examples/backtrack.txt'

    program = Program(FILE)
    print(program.atoms)
    for i in program.atoms.keys():
        print(program.atoms[i].name, program.atoms[i].fact, program.atoms[i].definition)

    program.to_problog(path='./examples/out.pl')
    program.to_dimacs(path='./examples/out.dimacs')