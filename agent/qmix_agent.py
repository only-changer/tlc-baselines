from . import RLAgent
import gym
import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class QMix_Agent(RLAgent):
    def __init__(self, action_space, state_size, ob_generator, reward_generator):
        super().__init__(action_space, ob_generator, reward_generator)

        self.action_size        = self.action_space.n
        self.state_size         = state_size
        self.id = ob_generator.iid

        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.learning_start     = 10
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.sample_batch_size  = 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def load_model(self, dir="examples/qmix_weights"):
        name = "agent_{}.h5".format(self.id)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="examples/qmix_weights"):
        name = "agent_{}.h5".format(self.id)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < self.sample_batch_size:
            return
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        for state, action, reward, next_state in sample_batch:
            state = np.reshape(state, [1,-1])
            next_state = np.reshape(next_state, [1,-1])
            target = reward + self.gamma*np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            #print('loss',history.history['loss'])
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        return self.reward_generator.generate()

    def get_value(self, ob):
        ob = np.reshape(ob, [1,-1])
        return self.model.predict(ob)

    def get_action(self, ob):
        if np.random.rand() <= self.exploration_rate:
            self.action = random.randrange(self.action_size)
        else:
            act_values = self.get_value(ob)
            self.action = np.argmax(act_values[0])
        return self.action