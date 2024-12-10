import pandas as pd
import numpy as np
from scipy.stats import chi2
import warnings
from joblib import Parallel, delayed
from collections import Counter
from math import log2


def read_from_file(file_path):
    return pd.read_csv(file_path)


def entropy_calculation(examples):
    label_counts = Counter(examples.iloc[:, -1])
    total_count = len(examples)
    entropy = 0.0

    for count in label_counts.values():
        probability = count / total_count
        entropy -= probability * log2(probability)

    return entropy


def split_by_attribute(data):
    for column in data.columns[:-1]:
        if column == "DEP_TIME_BLK":
            data[column] = data[column].apply(lambda x: "0000-1159" if int(x.split('-')[0]) < 1200 else "1200-2359")
        elif column == "MONTH":
            data[column] = data[column].apply(lambda x: "Jan-Jun" if x <= 6 else "Jul-Dec")
        elif column == "CONCURRENT_FLIGHTS":
            data[column] = data[column].apply(lambda x: "1-50" if x <= 50 else "51-109")
        elif column == "TMAX":
            data[column] = data[column].apply(lambda x: "-10-32" if x <= 32 else "33-115")
        elif data[column].dtype == object:
            data[column] = data[column].apply(lambda x: "A-M" if str(x)[0].upper() < 'M' else "M-Z")
        else:
            median = data[column].median()
            data[column] = data[column].apply(lambda x: f"-inf-{median}" if x <= median else f"{median}-inf")
    return data


def importance_information_gain(examples, attribute):
    total_entropy = entropy_calculation(examples)
    values = examples[attribute].unique()
    weighted_entropy = 0.0

    for value in values:
        subset = examples[examples[attribute] == value]
        subset_entropy = entropy_calculation(subset)
        weighted_entropy += (len(subset) / len(examples)) * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain


class NodeInDecisionTree:
    def __init__(self, attribute=None, value=None, results=None, branches=None, is_leaf=None, examples=None):
        self.attribute = attribute
        self.value = value
        self.results = results
        self.branches = branches or {}
        self.is_leaf = is_leaf if is_leaf is not None else False
        self.examples = examples
        self.median_value = None

    def determine_leaf_status(self):
        self.is_leaf = not bool(self.branches)


def plurality_value(examples, target_attribute):
    if examples.empty:
        return None
    target_values = examples[target_attribute]
    return target_values.mode()[0]


def decision_tree_learning(examples, attributes, parent_examples, target_attribute):
    if examples.empty:
        return NodeInDecisionTree(results=plurality_value(parent_examples, target_attribute), examples=parent_examples,
                                  is_leaf=True)
    elif len(examples[target_attribute].unique()) == 1:
        return NodeInDecisionTree(results=examples[target_attribute].iloc[0], examples=examples, is_leaf=True)
    elif not attributes:
        return NodeInDecisionTree(results=plurality_value(examples, target_attribute), examples=examples, is_leaf=True)

    # find the best attribute
    best_attribute = max(attributes, key=lambda attr: importance_information_gain(examples, attr))
    tree = NodeInDecisionTree(attribute=best_attribute, examples=examples)
    # copy examples to avoid modifying the original DataFrame
    examples = examples.copy()
    # get the unique values of the best attribute
    attribute_values = examples[best_attribute].unique()

    # iterate over the unique values to create subtrees
    for value in attribute_values:
        examples_of_bin_value = examples[examples[best_attribute] == value]
        subtree = decision_tree_learning(
            examples_of_bin_value,
            [attr for attr in attributes if attr != best_attribute],
            parent_examples,
            target_attribute)
        tree.branches[value] = subtree

    return tree


def print_decision_tree(node, depth=0, prefix=''):
    if node is None:
        return

    if not isinstance(node, NodeInDecisionTree):
        print(f"{prefix}Predicted: {node}")
        return

    if node.results is not None:
        if isinstance(node.results, pd.Series):
            most_common_value = node.results.mode()[0]
            print(f"{prefix}Predicted: {most_common_value}")
        else:
            print(f"{prefix}Predicted: {node.results}")
        return

    if depth == 0:
        print(f"{prefix}{node.attribute}")
    else:
        print(f"{prefix}|-- {node.attribute}")

    for branch_value, branch_node in node.branches.items():
        if isinstance(branch_value, pd.Interval):
            branch_value_str = f"({branch_value.left}, {branch_value.right}]"
        elif isinstance(branch_value, str):
            branch_value_str = f"'{branch_value}'"
        else:
            branch_value_str = str(branch_value)
        new_prefix = f"{prefix} |" if depth > 0 else f"{prefix}    "
        print(f"{new_prefix}{branch_value_str}:")
        print_decision_tree(branch_node, depth + 1, new_prefix + "  ")


def train_and_test_split(data, ratio):
    train_data = data.sample(frac=ratio, random_state=1)
    test_data = data.drop(train_data.index)
    return train_data, test_data


def divide_tree(tree):
    divided_data = []
    for branch in tree.branches.values():
        divided_data.append(branch.examples)
    return divided_data


def majority_count(examples):
    if isinstance(examples, pd.Series):
        late_flights = examples.sum()
        total_flights = len(examples)
        return late_flights / total_flights
    return 0


def prune_tree(tree, parent=None, bin_value=None):
    # if the current node is a leaf node, nothing to prune
    if tree.is_leaf:
        return tree

    # recursively prune branches
    for key, branch in tree.branches.items():
        if isinstance(branch, NodeInDecisionTree):
            prune_tree(branch, tree, key)

    # check if the current node needs to be pruned
    divided_data = divide_tree(tree)
    if chi_squared_test(tree, divided_data):
        # do not prune if chi-squared test is significant
        return tree

    # otherwise, accept h0, prune the node and replace it with the majority
    decision = 1 if majority_count(tree.results) > 0.5 else 0
    if parent is not None and bin_value is not None:
        parent.branches[bin_value] = decision
    else:
        tree.is_leaf = True
        tree.results = decision
        tree.branches = {}

    return tree


def chi_squared_test(tree, divided_data):
    p = 0.95
    statistic_value = 0
    # by h0:
    odds_to_late = majority_count(tree.results)
    odds_to_be_on_time = 1 - odds_to_late

    for i in range(len(tree.branches)):
        if len(divided_data[i]) == 0:
            continue
        estimate_p = len(divided_data[i]) * odds_to_late
        estimate_n = len(divided_data[i]) * odds_to_be_on_time
        actual_p = majority_count(divided_data[i]) * len(divided_data[i])
        actual_n = len(divided_data[i]) * (1 - majority_count(divided_data[i]))

        if estimate_p != 0:
            statistic_value += ((actual_p - estimate_p) ** 2) / estimate_p
        if estimate_n != 0:
            statistic_value += ((actual_n - estimate_n) ** 2) / estimate_n

    n = len(tree.examples)
    df = n - 1
    critic_value = chi2.ppf(p, df)
    return statistic_value <= critic_value


def check_majority(examples):
    sum_late = examples.iloc[:, -1].sum()  # sum of the target attribute
    sum_not_late = len(examples) - sum_late
    return sum_late / (sum_late + sum_not_late)


def predict_with_tree(tree, instance):
    if tree.is_leaf:
        return tree.results

    attribute_value = instance[tree.attribute]

    if attribute_value in tree.branches:
        return predict_with_tree(tree.branches[attribute_value], instance)
    else:
        # if the value is not found in the branches, return the plurality value
        return plurality_value(tree.examples, 'DEP_DEL15')


def calculate_error_rate(tree, test_data, target_attribute):
    incorrect_predictions = 0
    for _, instance in test_data.iterrows():
        prediction = predict_with_tree(tree, instance)
        actual_value = instance[target_attribute]
        # if the result of the tree doesn't match the actual result
        if prediction != actual_value:
            incorrect_predictions += 1
    total_instances = len(test_data)
    error_rate = incorrect_predictions / total_instances

    return error_rate


def start_building_tree(train_data):
    target_attribute = 'DEP_DEL15'
    attributes = [col for col in train_data.columns if col != target_attribute]
    decision_tree = decision_tree_learning(train_data, attributes, train_data, target_attribute)
    pruned_tree = prune_tree(decision_tree)
    return pruned_tree


def build_tree(ratio):
    data = read_from_file('flightdelay.csv')
    data = pd.DataFrame(data)
    data = split_by_attribute(data)
    target_attribute = 'DEP_DEL15'
    train_data, test_data = train_and_test_split(data, ratio)
    pruned_tree = start_building_tree(train_data)
    print_decision_tree(pruned_tree)
    # evaluate the pruned tree on the test data
    error_rate = calculate_error_rate(pruned_tree, test_data, target_attribute)
    print(f"Error rate: {error_rate:.2%}")


def divide_data_to_k(data, k):
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    divided_data = np.array_split(shuffled_data, k)
    return divided_data


def train_tree_on_fold(train_data, validation_data):
    pruned_tree = start_building_tree(train_data)
    error_rate = calculate_error_rate(pruned_tree, validation_data, 'DEP_DEL15')
    return error_rate


def tree_error(k):
    warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
    data = read_from_file('flightdelay.csv')
    divided_data = divide_data_to_k(data, k)

    results = Parallel(n_jobs=-1)(delayed(train_tree_on_fold)(
        pd.concat([divided_data[j] for j in range(k) if j != i]),
        divided_data[i]
    ) for i in range(k))

    average_error_rate = sum(results) / k
    print(f"Average error rate for {k}-fold cross-validation: {average_error_rate:.2%}")


def is_late(row_input):
    data = read_from_file('flightdelay.csv')
    attributes = data.columns.tolist()
    row_df = pd.DataFrame([row_input], columns=attributes[:-1])
    pruned_tree = start_building_tree(data)
    prediction = predict_with_tree(pruned_tree, row_df.iloc[0])
    print(f"Prediction for the input row: {prediction}")




