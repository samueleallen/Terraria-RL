import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_env import TerrariaEnv

# Create environment instance
env = DummyVecEnv([lambda: TerrariaEnv()])

model = PPO("MlpPolicy", env, verbose=1, device='cpu')

print("Starting training...")
model.learn(total_timesteps=100000)
print("Finished training.")

# Save the model
# model.save("ppo_terraria_bot")
# print("Model saved as ppo_terraria_bot.zip")daddaa