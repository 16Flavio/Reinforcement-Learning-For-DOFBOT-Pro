from typing import Optional
import numpy as np
import gymnasium as gym
import mujoco, mujoco.viewer
import numpy as np
import os
from pathlib import Path

class ReachEnv(gym.Env):

    def __init__(self):
        
        # Constante
        self.maxDecision = 50 # à 10 décisions à la seconde, ca fait 5 secondes de manipulation
        self.stepModel = 50 # 50 pas de MuJoCo = 100 ms entre chaque décision de l'agent
        self.maximumAngle = 5*(np.pi/180) # 5 degrés maximum par décision

        # Compteur
        self.decision = 0

        # Importation du modèle
        self.model = mujoco.MjModel.from_xml_path(os.path.join(Path(os.path.dirname(__file__)).parent.parent / "3DModels", "dofbot_pro.xml"))
        self.data = mujoco.MjData(self.model)

        # Positions initialisées seront défini de manière aléatoire dans reset()
        # On utilise -1,-1 en tant que "non-initialisé"
        self._target_location = np.array([-1, -1, -1], dtype=np.float32) # [x, y, z]

        # Définition de ce que l'agent observe
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        # [servos_1, servos_2, servos_3, servos_4] en radians
        # [x, y, z] coordonnées de la cible
        # Vecteur entre la position de la cible et la position du bout du bras [x, y, z]
        # Au total, 4 + 3 + 3 = 10 entrées

        # Définition des actions disponibles
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)   # [servos_1, servos_2, servos_3, servos_4]

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            Numpy Array: Observation with servos, target and vector from itself to target
        
        (Repris de la documentation officiel https://gymnasium.farama.org/introduction/create_custom_env/)
        """

        obs = []
    
        for angle in self.data.qpos[:4]:
            obs.append(angle)
        
        for coordinate in self._target_location:
            obs.append(coordinate)

        for i in range(3):
            obs.append(self._target_location[i] - self.data.site("grip_site").xpos[i])

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target

        (Repris de la documentation officiel https://gymnasium.farama.org/introduction/create_custom_env/)
        """
        return {
            "distance": np.linalg.norm(self._target_location - self.data.site("grip_site").xpos),
            "atteint": np.linalg.norm(self._target_location - self.data.site("grip_site").xpos) <= 0.01
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Commence un nouvel épisode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state

        (Repris de la documentation officiel https://gymnasium.farama.org/introduction/create_custom_env/)
        """

        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Remise à zéro des compteur
        self.decision = 0
        mujoco.mj_resetData(self.model, self.data)

        while True:
            # Place de manière aléatoire la cible à partir des angles
            angles_target_random = self.np_random.uniform(low=self.model.jnt_range[:4, 0], high=self.model.jnt_range[:4, 1], size=4)
            # On simule la position
            self.data.qpos[:4] = angles_target_random
            mujoco.mj_forward(self.model, self.data)

            if (self.data.ncon == 0) and (self.data.site("grip_site").xpos[2] >= 0) and (self.data.site("grip_site").xpos[1] >= 0.1):
                break

        # On récupère la position comme un point à atteindre
        self._target_location = self.data.site("grip_site").xpos.copy()

        while True:
            # On prend aléatoirement une nouvelle position
            angles_start_random = self.np_random.uniform(low=self.model.jnt_range[:4, 0], high=self.model.jnt_range[:4, 1], size=4)
            # On simule la nouvelle position
            self.data.qpos[:4] = angles_start_random
            self.data.ctrl[:4] = angles_start_random
            mujoco.mj_forward(self.model, self.data)

            if (self.data.ncon == 0) and (self.data.site("grip_site").xpos[2] >= 0) and (self.data.site("grip_site").xpos[1] >= 0.1):
                break

        # Retour standard de la bibliothèque
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)

        (Repris de la documentation officiel https://gymnasium.farama.org/introduction/create_custom_env/)
        """

        # On ramène l'action entre -1 et 1 (rien n'oblige l'agent de respecter ça)
        action = np.clip(action, -1, 1)

        # On récupère les angles en radians
        action = action*self.maximumAngle
        # On les bornes aux valeurs maximales
        angles_nouveau = np.clip(self.data.ctrl[:4] + action, self.model.jnt_range[:4, 0], self.model.jnt_range[:4, 1])

        # On les applique
        self.data.ctrl[:4] = angles_nouveau
        for i in range(self.stepModel):
            mujoco.mj_step(self.model, self.data)

        # Retour standard de la bibliothèque
        observation = self._get_obs()
        info = self._get_info()

        if np.linalg.norm(observation[-3:]) <= 0.01:
            reward = 1
        else:
            reward = 1/(info["distance"]*100 + 1e-6)

        self.decision += 1
        if self.decision >= self.maxDecision:
            truncated = True
        else:
            truncated = False

        if self.data.ncon >= 1:
            terminated = True
            reward = -10
        else:
            terminated = False

        return observation, reward, terminated, truncated, info