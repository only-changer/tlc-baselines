from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = ob_generator.iid

        self.yellow_phase_id = ob_generator.world.intersection_yellow_phase_id[self.iid]

        self.phase_list = [i for i in range(self.action_space.n)]

        self.action_size = len(self.phase_list)
        self.ob_space = ob_generator.ob_space
        if len(self.ob_space) == 1:
            self.ob_space = self.ob_space[0]
        self.ob_length = self.ob_space[0]

        self.memory = deque(maxlen=2000)
        self.learning_start = 300
        self.update_model_freq = 100
        self.update_target_model_freq = 1000

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.update_target_freq = 5
        self.batch_size = 30

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.last_action = self.phase_list[0]
        self.action = self.phase_list[-1]

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()[0]
        reward += (self.action == self.last_action) * 2
        self.last_action = self.action
        return reward

    def get_action(self, ob):
        self.action = self.phase_list[self.choose_action(ob)]
        return self.action

    def choose_action(self, ob):
        ob = self._reshape_ob(ob)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, input_dim=self.ob_length, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, [1, -1])

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        # action = self.phase_list.index(action)
        action = self.phase_list.index(action)
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for ob, action, reward, next_ob in minibatch:
            # print(next_state)
            ob = self._reshape_ob(ob)
            next_ob = self._reshape_ob(next_ob)
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_ob)[0]))
            target_f = self.model.predict(ob)
            target_f[0][action] = target
            history = self.model.fit(ob, target_f, epochs=1, verbose=0)
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)