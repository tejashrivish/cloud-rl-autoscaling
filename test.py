from environment.cloud_env import CloudAutoScalingEnv


env=CloudAutoScalingEnv("data/workload_sorted.csv")

state, _ = env.reset()
total_reward = 0

for _ in range(50):
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break

print("Sanity check reward:", total_reward)
print("Final state:", state)
