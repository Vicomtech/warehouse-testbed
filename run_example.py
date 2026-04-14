from environment.env_storehouse import Storehouse
import numpy as np

env = Storehouse(logging=False, random_start=True, max_steps=50)

state, _ = env.reset()
done = False
t = 0

while not done:
    print(f"\n--- STEP {t} ---")
    env.render()

    mask = env.get_available_actions()
    valid_actions = [i for i, m in enumerate(mask) if m]

    if not valid_actions:
        print("No valid actions available.")
        break

    action = np.random.choice(valid_actions)

    state, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {env.norm_action(action)} | Reward: {reward:.3f}")

    done = terminated or truncated
    t += 1