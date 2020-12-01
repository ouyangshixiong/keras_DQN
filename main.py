import os
# 允许重复加载动态链接库
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
import random
import numpy as np
from keras.optimizers import Adam
from algorithm import DQN

def train(env, algorithm, episode, batch):
        """训练
        Arguments:
            episode: 游戏次数
            batch： batch size

        Returns:
            history: 训练记录
        """
        algorithm.model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # 通过贪婪选择法ε-greedy选择action。
                x = observation.reshape(-1, 4)
                action = algorithm.sample(x)
                observation, reward, done, _ = env.step(action)
                # 将数据加入到经验池。
                reward_sum += reward
                algorithm.remember(x[0], action, reward, observation, done)

                if len(algorithm.memory_buffer) > batch:
                    # 训练
                    X, y = algorithm.process_batch(batch)
                    loss = algorithm.model.train_on_batch(X, y)

                    count += 1
                    # 减小egreedy的epsilon参数。
                    algorithm.update_epsilon()

                    # 固定次数更新target_model
                    if count != 0 and count % 20 == 0:
                        algorithm.update_target_model()

            if reward_sum == 200:
                break;

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
    
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, algorithm.epsilon))

        algorithm.model.save_weights('dqn.h5')

        return history

def test(env, model):
        """使用训练好的模型测试游戏.
        """
        observation = env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 5:
            env.render()

            x = observation.reshape(-1, 4)
            q_values = model.predict(x)[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = env.reset()

        env.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    algorithm = DQN()
    history = train(env, algorithm, 600, 32)
    test(env, algorithm.model)