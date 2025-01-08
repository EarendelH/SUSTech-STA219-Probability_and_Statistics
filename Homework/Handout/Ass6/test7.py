import numpy as np

rows, cols = 20, 50
total_trees = rows * cols

prob_left = 0.8
prob_above = 0.3

num_simulations = 38416

def simulate_fire():
    forest = np.zeros((rows, cols), dtype=bool)
    forest[0, 0] = True  

    for i in range(rows):
        for j in range(cols):
            if i > 0 and forest[i-1, j] and np.random.rand() < prob_above:
                forest[i, j] = True
            if j > 0 and forest[i, j-1] and np.random.rand() < prob_left:
                forest[i, j] = True

    return np.sum(forest)

burned_trees = np.array([simulate_fire() for _ in range(num_simulations)])

threshold = 0.3 * total_trees
probability = np.mean(burned_trees > threshold)

mean_burned_trees = np.mean(burned_trees)
std_burned_trees = np.std(burned_trees)

print(f"超过30%树木燃烧的概率: {probability:.5f}")
print(f"受影响的树木总数的预测值: {mean_burned_trees:.2f}")
print(f"受影响的树木总数的标准差: {std_burned_trees:.2f}")