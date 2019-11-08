import tensorflow as tf
from time import time
import numpy as np
from itertools import product

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
from tqdm import tqdm
import tensorflow_probability as tfp
tfd = tfp.distributions

import gym


#env_name = ''
env_name = 'BipedalWalker-v2'
# env_name = 'CartPole-v1'
def evaluate(model, n=1, disp=False):
    env = gym.make(env_name)
    #no_rew_early_stop = 250
    rewards = []
    for r in range(n):
        #self.rnn.reset_states()
        done = False
        obs = env.reset()
        last_rew = 0
        tot_rew = 0

        rews = []

        for i in range(50000):
            if done:
                break

            if disp:
                env.render()

            obs = obs.astype(np.float32)
            obs = obs.reshape((1, *obs.shape))

            dist, value = model(obs)

            action = dist.sample()
            action = list(np.array(action)[0])
            # action = np.argmax(action[0].numpy())


            obs, rew, done, _ = env.step(action)
            rews.append(round(rew, 2))

            tot_rew += rew
            if rew <= 0:
                last_rew += 1
            else:
                last_rew = 0

            # if len(rews) > 150:
            #     if sum(rews[-100:]) < 0:
            #         break


        rewards.append(tot_rew)
    env.close()
    return rewards



class ActorCritic(Model):

    def __init__(self, action_dim, **kwargs):
        super().__init__(**kwargs)

        self.critic = tf.keras.Sequential(
            [
                Dense(64),
                ReLU(),
                Dense(32),
                ReLU(),
                Dense(1)
            ]
        )
        
        self.actor = tf.keras.Sequential(
            [
                Dense(256),
                ReLU(),
                Dense(128),
                ReLU(),
                Dense(action_dim)
            ]
        )
        self.std = 1.

    def call(self, x):
        value = self.critic(x)
        mu = self.actor(x) #[0]
        # print(value)
        # print(mu)
        dist = tfd.Normal(scale=self.std, loc=mu) #?
        #....
        return dist, value


class ExperienceBuffer():
    
    def __init__(self, size=10000):
        self.size=size
        self.buffer = []
        
    def add(self, exp):
        self.buffer.append(exp)
    
    def sample(self, sample_size):
        return np.array(random.sample(self.buffer, k=sample_size))


def get_exp(env, batch_size=3000):

    log_probs = []
    values = []
    observations = []
    actions = []
    rewards = []
    masks = []


    entropy = 0

    done = False

    obs = env.reset()
    obs = obs.astype(np.float32)
    obs = obs.reshape((1, *obs.shape))
    print("START")

    steps = 0
    while True:

        dist, value = model(obs)
        #env.render()

        action = dist.sample()
        # action = np.argmax(action[0].numpy())
        action = list(np.array(action)[0])

        try:        
            next_obs, rew, done, _ = env.step(action)
        except:
            print(dist.sample())
            print(action)
            print(actions)
            print(obs)
            raise
        
        next_obs = next_obs.astype(np.float32)
        next_obs = next_obs.reshape((1, *next_obs.shape))

        log_prob = dist.log_prob(action)

        entropy += tf.math.reduce_mean(dist.entropy())
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(rew)
        masks.append((1 - done))
        
        observations.append(obs)
        actions.append(action)
        
        obs = next_obs
        if done:
            obs = env.reset()
            obs = obs.astype(np.float32)
            obs = obs.reshape((1, *obs.shape))

        if steps > batch_size:
            _, value = model(next_obs)
            values += [value]
            values = np.array(values)
            log_probs = np.array(log_probs)
            actions = np.vstack(actions)
            observations = np.array(observations)
            returns, advantage = compute_advantage(rewards, masks, values)
            mini_batch_size  = 30

            train(5, mini_batch_size, observations, actions, log_probs, returns, advantage)

            log_probs = []
            values = []
            observations = []
            actions = []
            rewards = []
            masks = []

            entropy = 0

            steps = 0

            eval_res = evaluate(model, disp=True)
            print(eval_res)

        steps += 1

    
def prep_data(mini_batch_size, observations, actions, log_probs, returns, advantages):

    batch_size = len(observations)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)

        yield observations[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]



def train(ppo_epochs, mini_batch_size, observations, actions, log_probs, returns, advantages):
    clip_ratio = 0.2
    for _ in range(ppo_epochs):

        for obs, action, old_log_probs, return_, advantage in prep_data(mini_batch_size, observations, actions, log_probs, returns, advantages):
            with tf.GradientTape() as tape:
                # print('obs', obs)
                dist, value = model(obs)
                entropy = tf.math.reduce_mean(dist.entropy())
                # print('act', action)
                # print('dist', dist.sample())
                new_log_probs = dist.log_prob(action)
                # print('new_log_probs', new_log_probs)
                ratio = tf.math.exp((new_log_probs - old_log_probs))
                # print('ratio', ratio)
                rat_x_adv = ratio * advantage
                min_adv = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage

                # print('1', surr1)
                # print('2', surr2)
                actor_loss = - tf.math.reduce_mean(tf.minimum(rat_x_adv, min_adv))
                critic_loss = tf.math.reduce_mean(tf.square(return_ - value))
                # print(actor_loss, critic_loss)
                # print(entropy)
                loss = 0.5 * critic_loss + actor_loss - 0.005 * entropy
                # raise

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def compute_advantage(rewards, masks, values, gamma=0.99, lam=0.97): #GAE
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns = [gae + values[step]] + returns
    returns = np.array(returns)
    advantage = returns - np.array(values[:-1])

    return returns, advantage


if __name__=='__main__':
    optimizer = tf.keras.optimizers.Adam(3e-4)
    model = ActorCritic(4)
    env = gym.make(env_name)
    env.reset()


    epochs = 1000
    for e in range(epochs):
        print(e)
        get_exp(env)
        if not (e+1)%10:
            evaluate(model, env, disp=True)






