from gymnasium.envs.registration import register

register(
    id="shower_env/ShowerEnv-v0",
    entry_point="shower_env.envs:ShowerEnv",
)
