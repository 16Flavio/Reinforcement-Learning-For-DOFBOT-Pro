from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np


"""
PPO algorithm : 

    - The initial problem is the following : An agent interact with its environment. It observes a stat, chooses and action, receives a reward 
    and the goal is to maximize the sum of the rewards. It's policy \pi(a|s) gives the probability of each action given a known given state
    - Gradient policy : we make the agent plays, then we increase the probability of actions that lead to goo results and we dim thoses who lead to bad results.
    The warning here is that a big update could destroy the policy, the agent would play randomly.
    - PPO limits at each updates the policy update. It's the clipping goal. 

Key notions : 
    - Actor (neural network) : it's the policy. It takes the observation and gives back the gaussian mean on the actions.
    - Critic (neural network) : it estimates the V(s) : "knowing the stat, what sum of the futur rewards could we get ?". Gives a number
    - Reward-to-go : For each time step k, it's the sum of the futur rewards of the episode, pondered by G_k = r_k + \gamma r_{k+1} + \gamma^2 r_{k+2}+...
    Gamma works so that the nearest reward count for more than the distant ones. We are only watching the futur ones because an action cannot interfere with the past reward.
    - Avantage : A_k = G_k - V(s_k). We compare what was actually obtained with what the critic expected. if A > 0 : the action was better than expected, else it's worse so we discourage it.
    It's normalized (mean : 0, std : 1) to have a stable learning.
    - Ratio : r = \pi_new(a|s)/\pi_old(a|s) = 1 if the policy hasn't changed. We calculate it with the log_probabilities : r = exp(log(\pi_new) - log(\pi_old))

Clip objective : 
    L = min(r A, clip(r, 1-\epsilon,1+\epsilon)A)
    - If A > 0 : we want to increase r, but the gradiant stops when r > 1.2
    - If A < 0 : we wand to decrease r, but the gradient stops when r < 0.8
    We minimize -L.

Algorithm loop : 
    1) Collect a batch of trajectory with the actual policy (rollout)
    2) Calculate the rewards-to-go
    3) Calculate the advantages with the critic
    4) Make several updates of the actor and the critic on this same batch
    5) Throw away the batch and start over with a new policy
"""

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        """
        Retrieve the dim of the gymnasium environment
        Creates the actor, the critic, and an Adam optimizer
        Creates the Gaussian covarience matrix that controls the exploration (the greater it is, the scatter the actions are around the mean)
        """
        #Initialize the hyperparameters
        self._init_hyperparameters(hyperparameters)

        #Extracting environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #First step : input is initial policy parameters and initial value function parameters
        #initialization of actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        #initialize actor and critic networks
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        #create a variable for the matrix with an arbitrary stdev
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, obs):
        """
        The actor calculates the mean of the action
        The Multivariate Normal law is built
        An action is pulled randomly in this law and its log-prod is computed
        The action is sent in numpy for the environment, and it's log-prob is also sent. 
        """
        #query the actor network for a mean action, it's the same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        #Creating the mutlivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        #Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        #return the sampled action and the log prob of that action
        #detach() allows to detach graphs, since action and log_prob are tensions with computation graphs, so we 
        #get rid of the graph and just convert the action to a numpy array. 
        return action.detach().numpy(), log_prob.detach()

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialization of the hyperparameters
        """

        #Here are all the parameters to be changed later
        #They are set to a random number, it has to be changed later
        self.timesteps_per_batch = 4000
        self.max_timesteps_per_episode = 50
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 #recommended for the clip threshold
        self.lr = 0.005 #learning rate of optimizers
        self.save_freq = 10 #saving each 10 batch

        for param, val in hyperparameters.items():
            setattr(self, param, val)

    def rollout(self):
        """
        Init of the environment, then an episode is played step by step
        The observations, actions, log-probs and rewards are stocked
        """
        #Rollout function to collect our data

        #Batch data
        batch_obs = [] #observations : (number of timesteps per batch, dimension of observation)
        batch_acts = [] #actions : (number of timesteps per batch, dimension of action)
        batch_log_probs = [] #log probs of each action : (number of timesteps per batch)
        batch_rews = [] #rewards : (number of episodes, number of timesteps per episode)
        batch_rtgs = [] #rewards to go (number of timesteps per batch)
        batch_lens = [] #episodic lengths in batch : (number of episodes)

        batch_final_dists = []   # distance à la cible au dernier pas de chaque épisode
        batch_min_dists = []     # distance minimale atteinte pendant chaque épisode
        batch_reached = []       # l'épisode a-t-il atteint la cible (<= 1 cm) ?

        t = 0 #Number of timesteps run so far this batch
        while t < self.timesteps_per_batch:
            #incrementing t_so_far
            ep_rews = []

            obs, _ = self.env.reset()
            done = False
            min_dist = np.inf
            reached = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 #Incrementing timesteps ran this batch so far

                #collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                min_dist = min(min_dist, info["distance"])
                reached = reached or bool(info["atteint"])

                #Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            #Collect episodic length and rewards
            batch_final_dists.append(info["distance"])
            batch_min_dists.append(min_dist)
            batch_reached.append(reached)
            batch_lens.append(ep_t + 1) #plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
        stats = {
            "rews": batch_rews,
            "final_dist": batch_final_dists,
            "min_dist": batch_min_dists,
            "reached": batch_reached,
        }

        #reshaping data as tensors in the shape specified before returning
        #batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        #batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        #batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float) #super slow
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.stack(batch_log_probs).float()  

        #Step 4 : computing the rewards to go
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, stats

    def compute_rtgs(self, batch_rews):
        """
        rewards to go formula : G(s_k) = \sum_{i=k}^{T} \gamma^{i-k} R(s_i)
        G = reward to go function
        s_k = observation at timestep k
        T = timesteps per episode
        \gamma = discount factor
        R(s_i) = reward given some observation s_i

        For each episod, the rewards are looked over but in decreasing order to compute G_k in only one step
        """
        #The reward to go per episode per batch to return
        batch_rtgs = []
        #iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 #discount reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward*self.gamma
                batch_rtgs.insert(0, discounted_reward)
        #convert the rewards to go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def learn(self, total_timesteps):
        """
        Collecting loop, V computation, avantages computation, updates with the ratio, clip of the losses, backpropagation on the actor and the critic
        """
        t_so_far = 0 #timesteps simulated so far
        i_so_far = 0

        #Step 2 : doing a loop over k
        while t_so_far < total_timesteps:
            #algo step 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, stats = self.rollout()
            #calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            avg_len = np.mean(batch_lens)
            avg_ret = np.mean([np.sum(ep) for ep in stats["rews"]])
            avg_final = np.mean(stats["final_dist"])
            avg_min = np.mean(stats["min_dist"])
            success = np.mean(stats["reached"]) * 100

            print(
                f"Batch {i_so_far} | timesteps: {t_so_far} | "
                f"longueur: {avg_len:.1f} | retour: {avg_ret:.2f} | "
                f"dist. finale: {avg_final*100:.1f} cm | dist. min: {avg_min*100:.1f} cm | "
                f"succès: {success:.0f}%",
                flush=True,
            )

            #Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            #step 5 : calculate advantage
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) #Normalization of advantages

            for _ in range(self.n_updates_per_iteration):
                #calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                #calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                #calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                #calculate actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                #calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                #calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def evaluate(self, batch_obs, batch_acts):
        """
        Make a link with the batch and the actual network
        The critics gives the V for each observation, the actor gives a new mean \pi_new
        """
        #Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()
        #squeeze() changes the dimensionality of a tensor
        #here batch_obs has the following form : (timesteps per batch, dimension of observation)
        #and after passing through the critic function, the form will be :(timesteps per batch,1)
        #squeeze() will allow to have the following form : (timestep per batch)

        #Calculate the log prob of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs