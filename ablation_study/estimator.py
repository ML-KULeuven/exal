
import random
from statistics import mean

SIZE = 100
BOUND = 1000
SAMPLES = 500
TRIALS = 10000

error_std = []
error_set = []
for _ in range(TRIALS):
    function = [random.randrange(BOUND) for _ in range(SIZE)]
    exact = mean(function)

    samples_std = [function[sample] for sample in [random.randrange(SIZE) for _ in range(SAMPLES)]]
    samples_set = [function[sample] for sample in {random.randrange(SIZE) for _ in range(SAMPLES)}]

    error_std.append((mean(samples_std) - exact) ** 2)
    error_set.append((mean(samples_set) - exact) ** 2)

print(f"error_std: {mean(error_std)}")
print(f"error_set: {mean(error_set)}")
