import time
def _log_summary(ep_len, ep_ret, ep_num):
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
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, viewer):
	while True:
		obs, _ = env.reset()
		done = False
		t = 0
		ep_ret = 0

		while not done:
			t += 1

			action = policy(obs).detach().numpy()
			obs, rew, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			ep_ret += rew

			# Affichage : on synchronise le viewer avec l'état MuJoCo de ton env
			if viewer is not None:
				viewer.sync()
				time.sleep(0.1)   # 1 décision = 0,1 s simulée, donc temps réel

		yield t, ep_ret

def eval_policy(policy, env, render=False):
	if render:
		import mujoco.viewer
		with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
			for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, viewer)):
				_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
				if not viewer.is_running():   # fenêtre fermée : on arrête
					break
	else:
		for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, None)):
			_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)