#!/usr/bin/env python3
#coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy


# Total number of nodes in the network
N = 10_000
# Number of round for a single simulation.
NB_ROUND = 100
# Fixed n for EC (n, k)
EC_N = 120

# Range for k in EC(n, k).
EC_K_MIN = 4
EC_K_MAX = 40

# Reliability thresholds above which a node is out.
#
# Higher means that we are more tolerant on the reliability (a node will be
# considered out less frequently).
THRESHOLDS = {
    'high': 10,   # high reliability
    'medium': 5,  # medium reliability
    'low': 1      # low reliability
}

LEVY_MAX = levy.ppf(0.99)

# Generate random numbers between [0; limit], following a Lévy distribution.
def bounded_levy(size, limit=10):
    return [
        min(val, LEVY_MAX) * limit / LEVY_MAX
        for val in levy.rvs(size=size)
    ]

# Run a simulation for a given reliability threshold.
#
# For each `k` in `k_range` we run a simulation during NB_ROUND.
# At the end of each simulation, we count how many round had reliability issues
# (i.e at least one of the k-node that hold the data fragment is unavailable).
#
# A node is unavailable when it's score goes above the threshold.
# The score of each node is increased by a random amount (following a Lévy
# distribution) at each simulation round.
#
# When a node becomes unavailable, its score is reseted (we consider that a
# machine is not out more than one round).
def simulate(threshold):
    # Generate an initial score for the N nodes for the network.
    nodes_qos = [abs(n) for n in np.random.normal(size=N)]
    nodes_qos.sort()
    results = []
    for ec_k in range(EC_K_MIN, EC_K_MAX + 1):
        rounds = []
        print('simulation for threshold={}/k={}'.format(threshold, ec_k))
        for _ in range(NB_ROUND):
            # Adjust the nodes' score accoroding to a Lévy distribution.
            for i, delta in enumerate(bounded_levy(size=1000)):
                nodes_qos[i] += delta
            # "Churn"
            # - count the node that are above the threshold in the k-first nodes
            # - reintroduce them for the next round with a normal random QoS
            count = 0
            for i in range(ec_k):
                if nodes_qos[i] < threshold:
                    continue
                count += 1
                nodes_qos[i] = np.random.normal()
            rounds.append(count != 0)
        results.append(rounds.count(True) * 100 / NB_ROUND)
    return results


def plot(results):
    x = np.arange(EC_K_MIN, EC_K_MAX + 1, 1)
    for result in results:
        plt.plot(x, result['data'],
                 label='{} reliability'.format(result['threshold']))
    plt.ylim((0, 100))
    plt.xlim((EC_K_MIN, EC_K_MAX))
    plt.title('Evolution of the reliability')
    plt.xlabel('EC k parameter')
    plt.ylabel('percent of round with missing data fragment')
    plt.legend(loc='upper left')
    plt.show()


def main():
    results = []
    for name, value in THRESHOLDS.items():
        results.append({'threshold': name, 'data': simulate(value)})
    plot(results)

if __name__ == '__main__':
    main()
