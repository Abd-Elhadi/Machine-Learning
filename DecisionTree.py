import numpy as np
import pandas as pd


# Tree's node representation
class Node:
    def __init__(self, lbl):
        self.label = lbl
        self.nxt = {}

    def predict(self, x):
        if x[self.label] not in self.nxt:
            return

        Next = self.nxt[x[self.label]]
        if not isinstance(Next, Node):
            return Next
        return Next.predict(x)


def splitData(dataSet, train_ratio):
    train = dataSet.sample(frac=train_ratio)
    test = dataSet.drop(train.index)
    return train, test


# calculate entropy
def entropy(y):
    elements, counts = np.unique(y, return_counts=True)
    _entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return _entropy


# calculate information gain
def informationGain(feature, y):
    children, details = np.unique(feature, return_inverse=True)

    avg = 0
    for k in range(len(children)):
        child = y[details == k]
        entropy_child = entropy(child)

        avg += entropy_child * len(child) / len(y)

    return entropy(y) - avg


# recursively build the decision tree
def build_tree(x, y, features):
    unique, cnt = np.unique(y, return_counts=True)
    if len(unique) <= 1:
        return unique[0], 1
    if len(features) == 0:
        return unique[np.argmax(cnt)], 1

    gains = [informationGain(x[feat], y) for feat in features]
    optimal = features[np.argmax(gains)]
    node = Node(optimal)
    leaves_size = 0
    for choice in np.unique(x[optimal]):
        subset = x[optimal] == choice
        node.nxt[choice], tmp_leave_size = build_tree(x[subset], y[subset], [v for v in features if v != optimal])
        leaves_size += tmp_leave_size

    return node, (leaves_size + 1)


def generate_DT(x, y):
    return build_tree(x, y, features=x.columns.tolist())


# Fills missing data points by majority of the row
def fill_in_unknowns(dataSet, majority):
    x = dataSet.iloc[:, 1:]
    for j in range(x.shape[1]):
        x.iloc[:, j].replace('?', 'y' if majority[i] else 'n', inplace=True)


# find the majority of the votes for each issue of the 16 issues
def find_majority(dataSet):
    x = dataSet.iloc[:, 1:]
    return (x.isin(['y']).sum(axis=0) >= x.isin(['n']).sum(axis=0)).tolist()


# calculate accuracy
def accuracy(tree, df):
    total = 0
    for i in range(len(df)):
        if tree.predict(df.iloc[i, 1:]) == df.iloc[i, 0]:
            total += 1
    return total / (len(df) * 1.0)


# build the model and return the accuracies and tree sizes
def model(df, train_ratio, iterations, fill_in=False):
    train_accuracies = []
    test_accuracies = []
    tree_sizes = []

    for i in range(iterations):
        train, test = splitData(df, train_ratio)
        if fill_in:
            majority = find_majority(train)
            fill_in_unknowns(train, majority)
            fill_in_unknowns(test, majority)
        tree, tree_size = generate_DT(train.iloc[:, 1:], train.iloc[:, 0])

        train_accuracies.append(accuracy(tree, train))
        test_accuracies.append(accuracy(tree, test))
        tree_sizes.append(tree_size)

        # print('Training accuracy {:.2f}%'.format(100 * train_accuracies[i]))
        # print('Testing  accuracy {:.2f}%'.format(test_accuracies[i] * 100))
        # print("Tree size: " + str(tree_size))

    return train_accuracies, test_accuracies, tree_sizes


if __name__ == '__main__':
    dataSet = pd.read_csv('house-votes-84.data.txt', header=None)
    iterations = 5
    ratios = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    ratios_size = len(ratios)

    for i in range(ratios_size):
        print("For Training Ratio: {:.2f}%".format(ratios[i] * 100))
        print('-----------------------')
        train_accuracies, test_accuracies, tree_sizes = model(dataSet, ratios[i], iterations,
                                                              fill_in=False if i == 0 else True)
        print('Minimum train accuracy: {:.2f}%'.format(100 * np.min(train_accuracies)))
        print('Mean train accuracy: {:.2f}%'.format(100 * np.mean(train_accuracies)))
        print('Maximum train accuracy: {:.2f}%'.format(100 * np.max(train_accuracies)))

        print('\nMinimum test accuracy: {:.2f}%'.format(100 * np.min(test_accuracies)))
        print('Mean test accuracy: {:.2f}%'.format(100 * np.mean(test_accuracies)))
        print('Maximum test accuracy: {:.2f}%'.format(100 * np.max(test_accuracies)))

        print('\nMinimum tree size: ', np.min(tree_sizes))
        print('Mean tree size: ', np.mean(tree_sizes))
        print('Maximum tree size: ', np.max(tree_sizes))
        if i != 5:
            print("****************************\n")