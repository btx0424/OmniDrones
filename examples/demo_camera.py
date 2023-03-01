import torch
import hydra
import os

from typing import Dict
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots import RobotCfg
    from omni_drones.robots.drone import MultirotorBase
    from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
    from omni.isaac.core.prims import XFormPrim
    import omni.replicator.core as rep
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    import warp as wp
    from pxr import Gf, Sdf, Usd, UsdGeom
    from helpers import design_scene

    sim = SimulationContext(
        stage_units_in_meters=1.0, 
        physics_dt=0.005, rendering_dt=0.005, 
        sim_params=cfg.sim,
        backend="torch", device="cuda"
    )
    
    drones: Dict[str, MultirotorBase] = {}
    n = 3
    for i, model in enumerate([
        # "Crazyflie", 
        "Firefly", 
        # "Hummingbird",
        # "Neo11", 
    ]):
        drones[model] = MultirotorBase.REGISTRY[model](cfg=RobotCfg())
        translation = torch.zeros(n, 3)
        translation[:, 0] = i 
        translation[:, 1] = torch.arange(n)
        translation[:, 2] = 0.5
        drones[model].spawn(n, translation=translation)

    design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device=cfg.sim.device)
    camera.spawn("/World/envs/env_0/Firefly_0")

    prim_path = stage_utils.get_next_free_path("/World/Camera")
    cam1_prim = UsdGeom.Camera(prim_utils.define_prim(prim_path, "Camera"))
    cam1_prim_path = prim_utils.get_prim_path(cam1_prim)
    cam1_xform = XFormPrim(cam1_prim_path)
    rp1 = rep.create.render_product(cam1_prim_path, (1024, 512))
    rgb1 = rep.AnnotatorRegistry.get_annotator("rgb", device=cfg.sim.device)
    rgb1.attach([rp1])

    prim_path = stage_utils.get_next_free_path("/World/envs/env_0/Firefly_0/Camera")
    cam2_prim = UsdGeom.Camera(prim_utils.define_prim(prim_path, "Camera"))
    cam2_prim_path = prim_utils.get_prim_path(cam2_prim)
    cam2_xform = XFormPrim(cam2_prim_path)
    rp2 = rep.create.render_product(cam2_prim_path, (1024, 512))
    rgb2= rep.AnnotatorRegistry.get_annotator("rgb", device=cfg.sim.device)
    rgb2.attach([rp2])

    sim.reset()

    camera.initialize()

    for drone in drones.values():
        drone.initialize()

    for _ in range(4):
        sim.step(render=True)

    from collections import defaultdict
    from torchvision.io.video import write_video

    frames = defaultdict(list)
    i = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=not cfg.headless)
            continue
        for drone in drones.values():
            actions = drone.action_spec.zero((drone._count,))
            actions.fill_(0.)
            drone.apply_action(actions)
        sim.step()
        frames["env"].append(wp.to_torch(rgb1.get_data(device=cfg.sim.device)))
        frames["cam"].append(wp.to_torch(rgb2.get_data(device=cfg.sim.device)))
        print(cam1_xform.get_world_pose()[0])
        print(cam2_xform.get_world_pose()[0])
        camera.update(dt=0.0)
        print(camera)
        frames["rgb"].append(camera._data.output["rgb"])
        i += 1
        if i >= 100:
            break

    for k, v in frames.items():
        video_array = torch.stack(v).cpu()[..., :3]
        write_video("k.mp4", video_array, fps=24)

    simulation_app.close()

if __name__ == "__main__":
    main()