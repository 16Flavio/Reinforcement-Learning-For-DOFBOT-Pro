import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy
import time
import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium.vector import AutoresetMode
import env

"""
Code repris en partie de : https://medium.com/@jiawei.kong/reinforcement-learning-tutorial-soft-actor-critic-sac-f0a097b328c3
Replay buffer pour rejouer les anciennes transitions
"""
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 256, device: str = "cpu"):

        # Stockage en interne des paramètres
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.batch_size = batch_size
        self.current_size = 0
        self.ptr = 0
        self.device = device

        # Déclaration du buffer préalloué
        self.buffer ={
            "obs": np.zeros((size, self.obs_dim), dtype=np.float32),
            "acts": np.zeros((size, self.act_dim), dtype=np.float32),
            "rews": np.zeros((size, 1), dtype=np.float32),
            "next_obs": np.zeros((size, self.obs_dim), dtype=np.float32),
            "done": np.zeros((size, 1), dtype=np.float32),
        }

    def store(self, state, action, reward, next_state, done):
        """Stocke une transition dans le buffer"""  
        self.buffer["obs"][self.ptr] = state
        self.buffer["acts"][self.ptr] = action
        self.buffer["rews"][self.ptr] = reward
        self.buffer["next_obs"][self.ptr] = next_state
        self.buffer["done"][self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size  # Pointeur circulaire
        # Une fois le buffer plein, c'est la donnée la plus ancienne qui est enlevé
        self.current_size = min(self.current_size + 1, self.size)

    def sample_batch(self):
        """Tire des mini-batchs au hasard"""
        if self.current_size < self.batch_size:
            raise ValueError("Not enough samples in ReplayBuffer.")
        indices = np.random.randint(0, self.current_size, size=self.batch_size)
        return {key: torch.from_numpy(self.buffer[key][indices]).to(device=self.device) for key in self.buffer}

    def __len__(self):
        return self.current_size

"""
Code repris en partie de : https://medium.com/@jiawei.kong/reinforcement-learning-tutorial-soft-actor-critic-sac-f0a097b328c3
Le critique
"""
class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(obs_dim + act_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # Dimension de state (batch, obs_dim)
        # Dimension de action (batch, act_dim)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

"""
L'acteur
"""
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Actor, self).__init__()

        self.act_dim = act_dim

        self.hidden1 = nn.Linear(obs_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        # act_dim moyennes puis act_dim log(écart-type)
        self.out = nn.Linear(256, 2*act_dim)
    
    def forward(self, state: torch.Tensor, deterministic=False):
        # Dimension de state (batch, obs_dim)
        x = state
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        actions = self.out(x)
        # Dimension de actions (batch, 2*act_dim)
        means = actions[:, :self.act_dim]
        stds = torch.exp(torch.clamp(actions[:, self.act_dim:], -20, 2))

        if not deterministic:
            u = means + stds*torch.randn_like(stds)
        else:
            u = means
        a = torch.tanh(u)

        # log_prob de u, puis correction du tanh (changement de variable) : - log(1 - a²)
        # 1e-6 évite log(0) quand tanh arrondit à ±1 en float32
        # On soustrait par articulation avant de sommer, pour rester en (batch, act_dim) puis (batch, 1)
        log_prob = torch.distributions.Normal(means, stds).log_prob(u)
        log_prob = log_prob - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a, log_prob


class SAC():
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):

        # Constantes
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.target_entropy = -act_dim

        # Element necessaire pour entrainement via SAC
        self.A = Actor(obs_dim, act_dim).to(device)
        self.Q1 = Critic(obs_dim, act_dim).to(device)
        self.Q2 = Critic(obs_dim, act_dim).to(device)

        self.Q1_copy = copy.deepcopy(self.Q1)
        self.Q2_copy = copy.deepcopy(self.Q2)

        self.optimizer_Actor = torch.optim.Adam(self.A.parameters(), lr=3e-4)
        self.optimizer_Critic = torch.optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=3e-4)
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def update(self, batch):
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # On apprend log(alpha) mais on utilise alpha = exp(log_alpha), toujours positif
        # detach() : seules la perte d'alpha et son optimiseur peuvent modifier alpha
        alpha = self.log_alpha.exp().detach()

        # Update de la critique
        with torch.no_grad():
            next_act, next_log_prob = self.A(next_obs)
            
            out_Q1_copy = self.Q1_copy.forward(next_obs, next_act)
            out_Q2_copy = self.Q2_copy.forward(next_obs, next_act)
            
            y = rews + self.gamma * (1 - done) * (torch.minimum(out_Q1_copy, out_Q2_copy) - alpha*next_log_prob)
        
        out_Q1 = self.Q1.forward(obs, acts)
        out_Q2 = self.Q2.forward(obs, acts)

        loss_critic = F.mse_loss(out_Q1, y) + F.mse_loss(out_Q2, y)

        self.optimizer_Critic.zero_grad()

        loss_critic.backward()

        self.optimizer_Critic.step()

        # Update de l'acteur
        new_act, log_prob = self.A(obs)

        out_Q = torch.minimum(self.Q1(obs, new_act), self.Q2(obs, new_act))

        loss_actor = torch.mean(alpha * log_prob - out_Q)

        self.optimizer_Actor.zero_grad()

        loss_actor.backward()

        self.optimizer_Actor.step()

        # Update d'alpha
        loss_alpha = - torch.mean(self.log_alpha * (log_prob.detach() + self.target_entropy))

        self.optimizer_alpha.zero_grad()

        loss_alpha.backward()

        self.optimizer_alpha.step()

        # Update des copies
        with torch.no_grad():
            for p_copy, p in zip(self.Q1_copy.parameters(), self.Q1.parameters()):
                p_copy.copy_((1 - self.tau)*p_copy + self.tau*p)
            for p_copy, p in zip(self.Q2_copy.parameters(), self.Q2.parameters()):
                p_copy.copy_((1 - self.tau)*p_copy + self.tau*p)


        return loss_critic.item(), loss_actor.item(), self.log_alpha.exp().item()

    def act(self, obs, deterministic=False):
        # numpy (obs_dim,) pour une simulation, ou (n_envs, obs_dim) pour plusieurs en parallèle
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # Une seule observation : on ajoute la dimension de batch attendue par l'acteur
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            a, _ = self.A(obs, deterministic)
        a = a.cpu().numpy()
        # (1, act_dim) -> (act_dim,) pour une seule simulation
        return a[0] if single else a

    def save(self, path):
        torch.save({
            "actor": self.A.state_dict(),
            "Q1": self.Q1.state_dict(),
            "Q2": self.Q2.state_dict(),
            "Q1_copy": self.Q1_copy.state_dict(),
            "Q2_copy": self.Q2_copy.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.A.load_state_dict(checkpoint["actor"])
        self.Q1.load_state_dict(checkpoint["Q1"])
        self.Q2.load_state_dict(checkpoint["Q2"])
        self.Q1_copy.load_state_dict(checkpoint["Q1_copy"])
        self.Q2_copy.load_state_dict(checkpoint["Q2_copy"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])


def evaluate(environment, algo, episodes, render=False):
    returns, distances, successes = [], [], []

    viewer = None
    if render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(environment.model, environment.data)

    for _ in range(episodes):
        obs, info = environment.reset()
        episode_return, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = environment.step(algo.act(obs, deterministic=True))
            episode_return += reward
            done = terminated or truncated
            if viewer is not None:
                viewer.sync()
                # On ralentit pour voir le mouvement en temps réel (100 ms par décision)
                time.sleep(environment.stepModel * environment.model.opt.timestep)
        returns.append(episode_return)
        distances.append(info["distance"])
        successes.append(info["atteint"])

    if viewer is not None:
        viewer.close()

    return np.mean(returns), np.mean(distances), np.mean(successes)


def make_env():
    # Fonction au niveau du module (et pas une lambda) pour pouvoir être envoyée aux processus des simulations
    return env.ReachEnv()


def train(total_steps, warmup, n_envs, updates_per_step, log_every, eval_every, eval_episodes, save_path, seed):
    # Les simulations parallèles sont créées avant d'utiliser le GPU, pour que les processus n'héritent pas de CUDA
    # SAME_STEP : quand un épisode se termine, step() renvoie directement l'observation du nouvel épisode
    # et le vrai dernier état est rangé dans info["final_obs"]
    if n_envs == 1:
        envs = gym.vector.SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        envs = gym.vector.AsyncVectorEnv([make_env] * n_envs, autoreset_mode=AutoresetMode.SAME_STEP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {device} | {n_envs} simulation(s) en parallèle | {updates_per_step} mise(s) à jour par pas")

    np.random.seed(seed)
    torch.manual_seed(seed)

    eval_environment = env.ReachEnv()
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    envs.action_space.seed(seed)

    # Le buffer peut contenir tout l'entraînement : rien n'est jamais écrasé
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=total_steps + n_envs, device=device)
    algo = SAC(obs_dim, act_dim, device=device)

    # Chaque simulation reçoit sa propre graine (seed, seed + 1, ...)
    obs, _ = envs.reset(seed=seed)
    eval_environment.reset(seed=seed + n_envs)

    # Suivi des épisodes terminés
    episode_returns = np.zeros(n_envs)
    returns, distances, successes = [], [], []
    loss_critic, loss_actor, alpha = float("nan"), float("nan"), 1.0
    best_success, best_distance = -1.0, float("inf")
    start = time.time()

    # step compte les transitions, toutes simulations confondues : un tour de boucle en ajoute n_envs
    step = 0
    while step < total_steps:
        previous_step = step
        step += n_envs

        # Actions aléatoires au début pour remplir le buffer avec des données variées
        if step <= warmup:
            actions = envs.action_space.sample()
        else:
            actions = algo.act(obs)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        done = terminated | truncated

        # Pour les simulations qui viennent de finir, next_obs est déjà le début de l'épisode suivant :
        # on stocke le vrai dernier état, sinon la transition relierait deux épisodes différents
        real_next_obs = next_obs.copy()
        for i in np.where(done)[0]:
            real_next_obs[i] = infos["final_obs"][i]

        # Seul terminated coupe le bootstrap
        for i in range(n_envs):
            buffer.store(obs[i], actions[i], rewards[i], real_next_obs[i], float(terminated[i]))

        episode_returns += rewards
        for i in np.where(done)[0]:
            returns.append(episode_returns[i])
            distances.append(infos["final_info"]["distance"][i])
            successes.append(infos["final_info"]["atteint"][i])
            episode_returns[i] = 0.0

        obs = next_obs

        if step > warmup:
            for _ in range(updates_per_step):
                loss_critic, loss_actor, alpha = algo.update(buffer.sample_batch())

        # step avance de n_envs à la fois : on regarde si on vient de franchir un multiple de log_every
        if step // log_every > previous_step // log_every and returns:
            # Moyenne sur les 20 derniers épisodes
            print(
                f"[{step:>7}] retour {np.mean(returns[-20:]):7.2f} | "
                f"distance finale {np.mean(distances[-20:]) * 100:5.1f} cm | "
                f"succès {np.mean(successes[-20:]) * 100:5.1f} % | "
                f"alpha {alpha:.3f} | "
                f"perte critique {loss_critic:.3f} | perte acteur {loss_actor:.3f} | "
                f"{step / (time.time() - start):.0f} pas/s"
            )

        if step // eval_every > previous_step // eval_every:
            mean_return, mean_distance, success = evaluate(eval_environment, algo, eval_episodes)
            print(f"  -> évaluation : retour {mean_return:.2f} | distance finale {mean_distance * 100:.1f} cm | succès {success * 100:.1f} %")

            algo.save(save_path.with_name(save_path.stem + "_last.pt"))
            # Meilleur modèle : d'abord le taux de succès, puis la distance en cas d'égalité
            if (success, -mean_distance) > (best_success, -best_distance):
                best_success, best_distance = success, mean_distance
                algo.save(save_path)
                print(f"  -> meilleur modèle sauvegardé dans {save_path}")

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC pour la tâche reach du DOFBOT-Pro")
    parser.add_argument("--steps", type=int, default=300_000, help="nombre total de décisions dans l'environnement")
    parser.add_argument("--warmup", type=int, default=5_000, help="pas aléatoires avant de commencer à apprendre")
    parser.add_argument("--n-envs", type=int, default=1, help="nombre de simulations en parallèle")
    parser.add_argument("--updates-per-step", type=int, default=None,
                        help="mises à jour SAC par tour de boucle (par défaut n_envs : une mise à jour par transition, comme avec une seule simulation)")
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent / "checkpoints" / "sac_reach.pt"))
    parser.add_argument("--eval", type=str, default=None, help="chemin d'un modèle à évaluer au lieu d'entraîner")
    parser.add_argument("--render", action="store_true", help="affiche le viewer MuJoCo pendant l'évaluation")
    args = parser.parse_args()

    if args.eval is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        environment = env.ReachEnv()
        algo = SAC(environment.observation_space.shape[0], environment.action_space.shape[0], device=device)
        algo.load(args.eval)
        environment.reset(seed=args.seed)
        mean_return, mean_distance, success = evaluate(environment, algo, args.eval_episodes, render=args.render)
        print(f"Retour {mean_return:.2f} | distance finale {mean_distance * 100:.1f} cm | succès {success * 100:.1f} %")
    else:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        updates_per_step = args.updates_per_step if args.updates_per_step is not None else args.n_envs
        train(args.steps, args.warmup, args.n_envs, updates_per_step, args.log_every, args.eval_every,
              args.eval_episodes, save_path, args.seed)