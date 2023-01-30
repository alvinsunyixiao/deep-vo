import argparse
import io
import pickle

import numpy as np
import quaternion
from PIL import Image
from tqdm import trange

import habitat_sim as hsim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from sim.habitat.viz import display_sample

from utils.params import ParamDict
from utils.pose3d import Pose3D, Rot3D

def mk_simple_cfg(scene_path: str, p: ParamDict) -> hsim.Configuration:
    # simulator backend
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    # agent
    agent_cfg = hsim.agent.AgentConfiguration()

    # RGB sensor
    rgb_sensor_spec = hsim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = hsim.SensorType.COLOR
    rgb_sensor_spec.resolution = [p.img_size_hw[0], p.img_size_hw[1]]
    rgb_sensor_spec.sensor_subtype = hsim.SensorSubType.PINHOLE
    rgb_sensor_spec.position = [0.0, p.sensor_height, 0.0]
    rgb_sensor_spec.hfov = p.img_hfov

    # depth sensor
    depth_sensor_spec = hsim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = hsim.SensorType.DEPTH
    depth_sensor_spec.resolution = [p.img_size_hw[0], p.img_size_hw[1]]
    depth_sensor_spec.sensor_subtype = hsim.SensorSubType.PINHOLE
    depth_sensor_spec.position = [0.0, p.sensor_height, 0.0]
    depth_sensor_spec.hfov = p.img_hfov

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return hsim.Configuration(sim_cfg, [agent_cfg])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to simulator params")
    parser.add_argument("-s", "--scene", type=str, required=True,
                        help="path to the scene file")
    parser.add_argument("--nav-mesh", type=str, default=None,
                        help="path to nav mesh for autonomous exploration")
    parser.add_argument("--viz", action="store_true",
                        help="specify to visualize generated data")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output path to store output")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)
    cfg = mk_simple_cfg(args.scene, params)
    sim = hsim.Simulator(cfg)

    nav_mesh_path = args.nav_mesh
    if args.nav_mesh is None:
        nav_mesh_path = args.scene.replace(".glb", ".navmesh")
    sim.pathfinder.load_nav_mesh(nav_mesh_path)

    sim.pathfinder.seed(params.random_seed)
    np.random.seed(params.random_seed)

    agent = sim.initialize_agent(0)
    data = []
    for i in trange(params.num_samples):
        # sample random position
        while True:
            nav_point = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.distance_to_closest_obstacle(
                    nav_point, params.obstacle_search_radius) == params.obstacle_search_radius:
                break

        # set agent pose
        agent_state = agent.get_state()
        agent_state.position = nav_point
        agent_state.rotation = np.quaternion(1., 0., 0., 0.)
        agent.set_state(agent_state)

        # set camera pose
        agent_state = agent.get_state()

        # EUS coordinate !!??
        # @see: https://aihabitat.org/docs/habitat-sim/habitat_sim.geo.html
        pyr = np.random.uniform([-params.perturb.pitch, -np.pi, -params.perturb.roll],
                                [params.perturb.pitch, np.pi, params.perturb.roll])
        R_yaw = quaternion.from_rotation_vector([0., pyr[1], 0.])
        R_pitch = quaternion.from_rotation_vector([pyr[0], 0., 0.])
        R_roll = quaternion.from_rotation_vector([0., 0., pyr[2]])
        R_all = R_roll * R_pitch * R_yaw
        for sensor in agent_state.sensor_states:
            agent_state.sensor_states[sensor].rotation = R_all
        agent.set_state(agent_state, infer_sensor_states=False)

        obs = sim.get_sensor_observations()
        if args.viz:
            display_sample(obs["color_sensor"], obs["depth_sensor"])

        # save to data
        color_img = Image.fromarray(obs["color_sensor"], mode="RGBA")
        color_img_bytes = io.BytesIO()
        color_img.save(color_img_bytes, format="PNG")

        world_R_sensor = Rot3D(np.array([R_all.x, R_all.y, R_all.z, R_all.w]))
        world_T_sensor = Pose3D(world_R_sensor, agent_state.sensor_states["depth_sensor"].position)

        data.append({
            "image_png": color_img_bytes.getvalue(),
            "depth": obs["depth_sensor"],
            "pose": world_T_sensor.to_storage().numpy(),
        })

    with open(args.output, "wb") as f:
        pickle.dump(data, f)
        print(f"Data saved to {args.output}")
