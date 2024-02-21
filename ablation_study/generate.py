
import random

def bottom_up(facts, rules, fan):
    identifier = 0
    program = "\n"
    for _ in range(facts):
        program += f"atom{identifier} << 0.5f\n"
        identifier += 1

    program += "\n"
    for _ in range(rules):
        body = {f"atom{random.randrange(identifier)}" for _ in range(fan)}
        rule = ", ".join([("-" if random.random() < 0.5 else "") + atom for atom in body])
        program += f"atom{identifier} << {rule}\n"
        identifier += 1

    program += f"\nq << -atom{identifier - 1}\n"
    return program

def top_down(fact_prob, rule_min, rule_max, old_prob, min_depth, max_depth):
    lines = ["q << atom0"]
    definition = 0
    identifier = 0
    while definition <= identifier:
        if random.random() < fact_prob or max_depth <= definition:
            lines.append(f"atom{definition} << 0.5f")
        else:
            body = set()
            for _ in range(random.randint(rule_min, rule_max)):
                if definition < identifier and random.random() < old_prob:
                    body.add(f"atom{random.randint(definition + 1, identifier)}")
                else:
                    identifier += 1
                    body.add(f"atom{identifier}")
            body = ", ".join([("-" if random.random() < 0.5 else "") + atom for atom in body])
            lines.append(f"atom{definition} << {body}")
        definition += 1
    return "\n".join(lines[::-1])

def branch(depth=1, degree=2, index=0):
    if depth <= 0:
        return f"x{index} << 0.5f"
    else:
        children = [degree * index + offset + 1 for offset in range(degree)]
        body = ", ".join(f"x{child}" for child in children)
        return "\n".join(branch(depth=depth-1, degree=degree, index=child) for child in children) + f"\nx{index} << {body}"

def bayesian(internal=1, external=1, degree=1, lookback=1):
    program = "\n"
    assignments = 1 << degree
    total = internal + external
    for index in range(total):
        if index < lookback or random.randrange(internal + external) < external:
            program += f"a{index} << 0.5f\n"
            external -= 1
        else:
            parents = [f"a{index - parent - 1}" for parent in random.sample(range(lookback), degree)]
            bodies = []
            for assignment in range(assignments):
                if random.random() < 0.5:
                    bodies.append(", ".join(parents[parent] if assignment >> parent & 1 == 0 else "-" + parents[parent]
                                            for parent in range(degree)))
            if len(bodies) == 0:
                bodies = [", ".join(parents[parent] if random.random() < 0.5 else "-" + parents[parent]
                                    for parent in range(degree))]
            elif len(bodies) == assignments:
                bodies.pop(random.randrange(len(bodies)))
            for parent, body in enumerate(bodies):
                program += f"a{index}b{parent} << {body}\n"
            program += f"a{index} << " + ", ".join(f"-a{index}b{parent}" for parent in range(len(bodies))) + "\n"
            internal -= 1
    program += f"q << a{total - 1}\n"
    return program
