import argparse

import DenseMoE
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import obtain_data

import BTree

import MoEInexactIndex


def run_exact_index_experiment(data, index, verbose=False):
    for key in data:
        index.insert(key)
    return index.size_if_full(), index.depth()


def run_inexact_index_experiment(data, index, lookup_pattern, lookup_beta_a=None, lookup_beta_b=None, insertion_lookup_interspersion='random_uniform', verbose=False, show_graph=False):
    """
    Runs an experiment: trains the index on possibly interspersed sequence of insertions and lookups of keys in a dataset; evaluates the index by looking up every key in the dataset; returns the squared loss between the looked-up positions and their true positions during training and evaluation; optionally, displays a plot of the true position and the looked-up position for each key.

    Arguments:
        data {numpy.float} -- Vector of key values in insertion order.
        index {Index} -- Index to train.
        lookup_beta_a {float} -- the a parameter of the Beta distribution governing the lookup pattern distribution in the sorted data. Will supersede lookup_pattern if and only if lookup_beta_b is also specified.
        lookup_beta_b {float} -- the b parameter of the Beta distribution governing the lookup pattern distribution in the sorted data. Will supersede lookup_pattern if and only if lookup_beta_a is also specified.
        lookup_pattern {string} -- Lookup pattern during training: 'most_recent' to always look up the most recently inserted key, 'sequential' to look up the earliest inserted key not already looked up, or 'random_uniform' to look up a random key that is already inserted. Ignored if both lookup_beta_a and lookup_beta_b are specified.
        insertion_lookup_interspersion {float} in [0, 1) -- Probability during training that a lookup occurs instead of an insertion. If lookup_beta_a and lookup_beta_b are specified, a datum may be drawn that is not already inserted in the index, in which case the lookup will be skipped.

    Keyword Arguments:
        verbose {bool} -- Whether to print the key and position for each trained and looked-up key (default: {False}).
        show_graph {bool} -- Whether to display a plot of the true position and the looked-up position against the value of each key (default: {True}).

    Returns:
        [int] -- Training squared loss between looked-up positions and true positions
        [int] -- Evaluation squared loss between looked-up positions and true positions
        [numpy.int_] -- Predicted position for each key in {data}
        [numpy.int_] -- Number of times each key in {data} was trained
    """

    # Training
    position_insert = 0
    position_lookup = 0
    training_squared_loss = 0

    argsorted_data = None
    if lookup_beta_a is not None and lookup_beta_b is not None:
        argsorted_data = np.argsort(data)

    train_count = np.zeros_like(data)
    while True:
        is_insert = np.random.random_sample() > insertion_lookup_interspersion
        if is_insert:
            # insertion
            if position_insert >= len(data):
                break
            position = position_insert
            position_insert += 1
        else:
            # lookup
            if position_insert == 0:
                continue
            if lookup_beta_a is not None and lookup_beta_b is not None:
                rank = int(np.random.beta(
                    lookup_beta_a, lookup_beta_b)*len(data))
                if rank < len(data):
                    position = argsorted_data[rank]
                else:
                    continue
            elif lookup_pattern == 'most_recent':
                position = position_insert - 1
            elif lookup_pattern == 'sequential':
                position = position_lookup
                position_lookup += 1
            elif lookup_pattern == 'random_uniform':
                position = np.random.randint(0, position_insert)
            else:
                continue
            if position >= len(data) or position >= position_insert:
                continue
        true_position = np.nonzero(np.argsort(
            data[:position_insert]) == position)[0].item()
        key = data[position].item()
        predicted_position = index.lookup(key, verbose)
        training_squared_loss += (predicted_position - true_position) ** 2
        index.train(key, true_position, is_insert, verbose)
        train_count[position] += 1

    # Evaluation
    evaluation_predicted_position = np.empty_like(data, dtype=np.int_)
    evaluation_squared_loss = 0
    for true_position, lookup_datum in enumerate(data):
        evaluation_predicted_position[true_position] = index.lookup(
            lookup_datum, verbose)
        evaluation_squared_loss += (
            evaluation_predicted_position[true_position] - true_position) ** 2

    if show_graph:
        data_argsort = np.argsort(data)
        true_scatter = plt.scatter(
            data[data_argsort],
            range(len(data)),
            s=train_count[data_argsort] * 10,
            alpha=0.5,
            label='True position'
        )
        predicted_scatter = plt.scatter(
            data[data_argsort],
            evaluation_predicted_position[data_argsort],
            marker='.',
            edgecolors=None,
            label='Predicted position'
        )
        plt.xlabel('Key')
        plt.ylabel('Position')
        legend_lines, legend_labels = true_scatter.legend_elements(
            prop="sizes",
            func=lambda s: s / 10,
            color=true_scatter.get_facecolor().flatten(),
            num=5)
        plt.legend(
            [
                *legend_lines,
                lines.Line2D(
                    [], [],
                    color=predicted_scatter.get_facecolor().flatten(),
                    marker='x',
                    linestyle='None')
            ],
            [*legend_labels, 'Predicted position'],
            title='True position; number of times key was trained')
        plt.show()

    return training_squared_loss, evaluation_squared_loss, evaluation_predicted_position, train_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')  # only used to identify standard output
    parser.add_argument('dataset_filename')
    parser.add_argument('--dataset_dtype_str')
    parser.add_argument('--dataset_md5')
    # Sort precedes shuffle
    parser.add_argument('--sort_data', action='store_true')
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--btree_node_capacity', type=int)
    parser.add_argument('--moe_index_units', type=int, default=20)
    parser.add_argument('--moe_index_n_experts', type=int, default=20)
    parser.add_argument('--moe_index_epochs', type=int, default=10)
    parser.add_argument('--moe_index_learning_rate', type=float, default=0.1)
    parser.add_argument('--moe_index_batch_size', type=int, default=10)
    parser.add_argument('--moe_index_decay', type=float, default=1e-6)
    parser.add_argument('--moe_index_cache_max_size', type=int, default=5)
    parser.add_argument('--moe_experiment_lookup_pattern',
                        default='random_uniform')
    parser.add_argument('--moe_experiment_lookup_beta_a', type=float)
    parser.add_argument('--moe_experiment_lookup_beta_b', type=float)
    parser.add_argument('--moe_experiment_insertion_lookup_interspersion',
                        type=float, default=0.9)
    parser.add_argument('--moe_experiment_show_graph', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = obtain_data.decompress_array(
        args.dataset_filename, args.dataset_md5, args.dataset_dtype_str, verbose=args.verbose)

    if args.sort_data:
        data = np.sort(data)
    elif args.shuffle_data:
        data = np.random.default_rng().shuffle(data)

    size_if_full = None
    depth = None
    if args.btree_node_capacity is not None:
        btree_index_params = {
            'node_capacity': args.btree_node_capacity
        }
        btree_index = BTree.BTree(**btree_index_params)
        size_if_full, depth = run_exact_index_experiment(
            data, btree_index, args.verbose)
        if args.verbose:
            print(btree_index)

    moe_index_params = {
        'units': args.moe_index_units,
        'n_experts': args.moe_index_n_experts,
        'epochs': args.moe_index_epochs,
        'learning_rate': args.moe_index_learning_rate,
        'batch_size': args.moe_index_batch_size,
        'decay': args.moe_index_decay,
        'cache_max_size': args.moe_index_cache_max_size
    }
    moe_index = MoEInexactIndex.MoEInexactIndex(**moe_index_params)

    moe_experiment_params = {
        'lookup_pattern': args.moe_experiment_lookup_pattern,
        'lookup_beta_a': args.moe_experiment_lookup_beta_a,
        'lookup_beta_b': args.moe_experiment_lookup_beta_b,
        'insertion_lookup_interspersion': args.moe_experiment_insertion_lookup_interspersion,
        'show_graph': args.moe_experiment_show_graph
    }
    training_squared_loss, evaluation_squared_loss, _, _ = run_inexact_index_experiment(
        data, moe_index, **moe_experiment_params, verbose=args.verbose)

    print(','.join(map(str, [
        args.run_id,
        size_if_full, depth,
        training_squared_loss, evaluation_squared_loss
    ])))


if __name__ == "__main__":
    main()
