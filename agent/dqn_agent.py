from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator):
        super().__init__(action_space)
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator

        self.delta_t = 10
        self.state_size = config['state_size']
        self.action_size = config['action_size']

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 5
        self.batch_size = 30

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.phase_space = action_space.n

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        return self.reward_generator.generate()

    def get_action(self, ob):
        return self.action_space.sample()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action)
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name="model/dqn/trafficLight-dqn.h5"):
        self.model.load_weights(name)

    def save_model(self, name="model/dqn/trafficLight-dqn.h5"):
        self.model.save_weights(name)