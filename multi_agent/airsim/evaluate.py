import supersuit as ss
from stable_baselines3 import PPO
import time

from scripts.airsim_env import AirSimDroneEnv   # ← 改成你环境文件的名字

# ----------------------------------------------------
# 1. Load trained model
# ----------------------------------------------------
MODEL_PATH = "saved_policy/best_model"   # ← 你训练生成的 .zip 文件名
model = PPO.load(MODEL_PATH)

# ----------------------------------------------------
# 2. Create environment EXACTLY like training
# ----------------------------------------------------
def make_env():
    env = AirSimDroneEnv(
        ip_address="127.0.0.1",
        image_shape=(84, 84, 3),
        input_mode="single_rgb",
        num_drones=1
    )

    # Wrap to vectorized env (must match training!)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env

env = make_env()

# ----------------------------------------------------
# 3. Reset environment
# ----------------------------------------------------
obs = env.reset()
done = False

print("\n===== Evaluation started =====")

# ----------------------------------------------------
# 4. Evaluation loop
# ----------------------------------------------------
step = 0
while True:
    step += 1
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    print(f"Step {step}: action={action}, reward={reward}")

    # if env terminates
    if done:
        print("Episode finished.")
        break

    # Optional: slow down evaluation
    time.sleep(0.1)

print("===== Evaluation finished =====")
env.close()
