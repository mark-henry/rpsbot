import tensorflow as tf
import numpy as np

class RPSModel:
    def __init__(self, history_length, learning_rate=0.001):
        self.history_length = history_length
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.num_actions = 3  # rock, paper, and scissors :)

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.history_length, 2))
        lstm_layer = tf.keras.layers.LSTM(128)(input_layer)
        dense_layer = tf.keras.layers.Dense(64, activation='relu')(lstm_layer)
        output_layer = tf.keras.layers.Dense(self.num_actions, activation='softmax')(dense_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model

    def predict(self, history):
        history = np.array(history)
        history = np.expand_dims(history, axis=0)
        return self.model.predict(history)

    def train(self, history, action):
        history = np.array(history)
        action = np.array(action)
        history = np.expand_dims(history, axis=0)
        action = np.expand_dims(action, axis=0)
        target = np.zeros((1, self.num_actions))
        target[0][action] = 1
        self.model.train_on_batch(history, target)