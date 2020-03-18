import tensorflow as tf


class TicTacToeModel(tf.keras.Model):
    def __init__(self, size, seed):
        super(TicTacToeModel, self).__init__()
        self.size2 = size*size
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))
        self.dense3 = tf.keras.layers.Dense(self.size2, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))
    
    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
