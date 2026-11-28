import time
import torch
import numpy as np
def _log_summary(ep_len, ep_ret, ep_num, info):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num+1} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"dist. finale: {info['distance'] * 100:.1f} cm | "f"{'SUCCÈS' if info['atteint'] else 'échec'}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, viewer, n_episodes):
	for _ in range(n_episodes):
		obs, _ = env.reset()
		done = False
		t = 0
		ep_ret = 0
		info = {}

		while not done:
			# Fenêtre fermée en plein épisode : on arrête tout
			if viewer is not None and not viewer.is_running():
				return

			t += 1
			with torch.no_grad():
				action = policy(obs).numpy()
			obs, rew, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			ep_ret += rew

			if viewer is not None:
				viewer.sync()
				time.sleep(0.1)  # 1 décision = 0,1 s simulée

		yield t, ep_ret, info

def eval_policy(policy, env, render=False, n_episodes=20):
    rets, dists, succes = [], [], []

    def run(viewer):
        for ep_num, (ep_len, ep_ret, info) in enumerate(rollout(policy, env, viewer, n_episodes)):
            _log_summary(ep_len, ep_ret, ep_num, info)
            rets.append(ep_ret)
            dists.append(info["distance"])
            succes.append(bool(info["atteint"]))

    if render:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            run(viewer)
    else:
        run(None)

    if rets:
        print("\n------------------ Résumé ------------------", flush=True)
        print(f"Épisodes : {len(rets)}", flush=True)
        print(f"Retour moyen : {np.mean(rets):.2f}", flush=True)
        print(f"Distance finale moyenne : {np.mean(dists) * 100:.1f} cm", flush=True)
        print(f"Taux de succès : {np.mean(succes) * 100:.0f} %", flush=True)