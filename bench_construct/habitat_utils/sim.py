import math
import random

import habitat_sim
from habitat_sim import Simulator as Sim


def make_sim_cfg(settings, agent_setting):
    # sim cfg
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.default_agent_id = settings["default_agent"]
    sim_cfg.allow_sliding = settings["allow_sliding"]
    sim_cfg.random_seed = settings["seed"]
    # sim_cfg.enable_physics = True
    # agent cfg
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space["turn_left"] = habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=agent_setting["turn_angle"])
    )
    agent_cfg.action_space["turn_right"] = habitat_sim.agent.ActionSpec(
        "turn_right",
        habitat_sim.agent.ActuationSpec(amount=agent_setting["turn_angle"]),
    )
    agent_cfg.action_space["move_forward"] = habitat_sim.agent.ActionSpec(
        "move_forward",
        habitat_sim.agent.ActuationSpec(amount=agent_setting["step_size"]),
    )
    agent_cfg.height = agent_setting["height"]
    agent_cfg.radius = agent_setting["radius"]
    sim_sensors = []

    if agent_setting["random_view"]:
        # random setting
        height = random.randint(360, 640)
        width = random.randint(360, 640)
        camera_height = random.random() * 1.2 + 0.5  # 0.5~1.7
        position = agent_setting["rgb_sensor"]["position"]
        position[1] = camera_height
        hfov = random.randint(60, 120)
        min_pitch, max_pitch = [-30, 10]
        if camera_height < 1:
            min_pitch = 0
        if camera_height > 1.0:
            max_pitch = 0
        pitch = math.radians(random.randint(min_pitch, max_pitch))
    else:
        # original setting
        height = agent_setting["rgb_sensor"]["height"]
        width = agent_setting["rgb_sensor"]["width"]
        position = agent_setting["rgb_sensor"]["position"]
        hfov = agent_setting["rgb_sensor"]["hfov"]
        pitch = math.radians(agent_setting["rgb_sensor"]["pitch"])

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [height, width]
    rgb_sensor_spec.position = position
    rgb_sensor_spec.hfov = hfov
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_sensor_spec.orientation = [pitch, 0, 0]
    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [height, width]
    depth_sensor_spec.position = position
    depth_sensor_spec.hfov = hfov
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.orientation = [pitch, 0, 0]

    # add sensors
    sim_sensors.append(rgb_sensor_spec)
    sim_sensors.append(depth_sensor_spec)
    agent_cfg.sensor_specifications = sim_sensors
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_nav_mesh_cfg(settings):
    nav_mesh_cfg = habitat_sim.NavMeshSettings()
    nav_mesh_cfg.set_defaults()
    nav_mesh_cfg.agent_height = settings["agent_height"]
    nav_mesh_cfg.agent_radius = settings["agent_radius"]

    # nav_mesh_cfg.agent_max_climb = settings["agent_max_climb"]
    return nav_mesh_cfg


class HabitatSimulator:
    def __init__(self, sim_setting, agent_setting) -> None:
        # initialize environment
        self.config = make_sim_cfg(sim_setting, agent_setting)
        self.simulator = Sim(self.config)
        # recompute navigation mesh
        self.nav_mesh_config = make_nav_mesh_cfg(sim_setting)
        self.simulator.recompute_navmesh(
            self.simulator.pathfinder, self.nav_mesh_config
        )
        # get agent
        self.agent = self.simulator.initialize_agent(sim_setting["default_agent"])
