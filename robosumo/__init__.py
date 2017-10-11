from gym.envs.registration import register


# ------------------------------------------------------------------------------
# RoboSumo environments
# ------------------------------------------------------------------------------

register(
    id='RoboSumo-Ant-vs-Ant-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['ant', 'ant'],
        'agent_densities': [13., 13.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Ant-vs-Bug-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['ant', 'bug'],
        'agent_densities': [13., 10.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Ant-vs-Spider-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['ant', 'spider'],
        'agent_densities': [13., 39.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Bug-vs-Ant-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['bug', 'ant'],
        'agent_densities': [10., 13.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Bug-vs-Bug-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['bug', 'bug'],
        'agent_densities': [10., 10.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Bug-vs-Spider-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['bug', 'spider'],
        'agent_densities': [10., 39.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Spider-vs-Ant-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['spider', 'ant'],
        'agent_densities': [39., 13.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Spider-vs-Bug-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['spider', 'bug'],
        'agent_densities': [39., 10.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)

register(
    id='RoboSumo-Spider-vs-Spider-v0',
    entry_point='robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['spider', 'spider'],
        'agent_densities': [39., 39.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
    },
)
