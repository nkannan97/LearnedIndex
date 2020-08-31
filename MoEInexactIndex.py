import DenseMoE
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class MoEInexactIndex():
    """
    An index backed by a mixture-of-experts neural network and a randomly evicted cache of recently trained mappings.
    """

    def __init__(self, units, n_experts, epochs, learning_rate, batch_size, decay, cache_max_size):
        """
        Arguments:
            units {int} -- Number of hidden units between the mixture-of-experts and the output.
            n_experts {int} -- Number of expert neural networks.
            epochs {int} -- Number of epochs for which to train the neural network during each call to train().
            learning_rate {float} -- Optimiser learning rate.
            batch_size {int} -- Batch size of each training epoch.
            decay {float} -- Optimiser decay.
            cache_max_size {int} -- Number of elements in the cache before eviction.
        """

        self.model = tf.keras.models.Sequential()
        self.model.add(DenseMoE.DenseMoE(
            units,
            n_experts,
            expert_activation='relu',
            gating_activation='softmax'))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.RMSprop(
                lr=learning_rate,
                decay=decay),
            metrics=['accuracy'])

        self.cache = []
        self.cache_max_size = cache_max_size
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, key, position, is_insert, verbose=False):
        """
        Trains the index on a new mapping between a key and a position. If this is a new mapping in the index ({is_insert} is True), finds all mappings in the cache that have a greater key, and increments their positions by one. If the cache is full, evicts one mapping at random.

        Arguments:
            key {float} -- The key of the mapping.
            position {int} -- The position of the mapping.
            is_insert {bool} -- Whether this mapping is a new mapping. If so, finds all mappings in the cache that have a greater key, and increments their positions by one

        Keyword Arguments:
            verbose {bool} -- Whether to print the key and position (default: {False}).
        """

        rng = np.random.default_rng()
        if verbose:
            print('train: ' + str(key) + ', ' + str(position))
        if is_insert:
            for i, (cache_key, cache_position) in enumerate(self.cache):
                if cache_key >= key:
                    self.cache[i] = (cache_key, cache_position + 1)
        if len(self.cache) >= self.cache_max_size:
            del self.cache[rng.integers(len(self.cache))]
        self.cache.append((key, position))
        self.model.fit(
            np.array([t[0] for t in self.cache]).reshape(-1, 1),
            np.array([t[1] for t in self.cache]),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0)

    def lookup(self, key, verbose=False):
        """
        Predicts the position of a given key.

        Arguments:
            key {float} -- Key for which to predict position.

        Keyword Arguments:
            verbose {bool} -- Whether to print the key and predicted position (default: {False}).

        Returns:
            [int] -- Predicted position.
        """

        predicted_position = int(round(
            self.model.predict(np.reshape(key, (1, 1)),
                               batch_size=1).item()))
        if verbose:
            print('lookup: ' + str(key) + ', ' + str(predicted_position))
        return predicted_position
