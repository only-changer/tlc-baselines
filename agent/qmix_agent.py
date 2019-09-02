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
    def __init__(self, action_space, action_size, state_size, ob_generator, reward_generator):
        super().__init__(action_space, ob_generator, reward_generator)

        self.weight_backup      = "examples/qnet_weight.h5"
        self.action_size        = action_size
        self.state_size         = state_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.sample_batch_size  = 32
        self.model              = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_ob(self):
        raw_ob = self.ob_generator.generate()
        self.ob = raw_ob
        for i in range(self.state_size - len(raw_ob)):
            self.ob.append(0)
        self.ob = np.reshape(self.ob, [1, self.state_size])
        return self.ob

    def get_reward(self):
        return self.reward_generator.generate()

    def get_value(self, ob):
        self.act_values = self.model.predict(ob)
        return self.act_values

    def get_action(self, ob):
        if np.random.rand() <= self.exploration_rate:
            self.action = random.randrange(self.action_size)
        else:
            act_values = get_value()
            self.action = np.argmax(act_values[0])
        return self.action