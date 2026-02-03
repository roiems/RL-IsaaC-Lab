print("--------------------------------------------------")
print(">>> DEBUG: JUMP REWARDS FILE IS BEING LOADED <<<")
print("--------------------------------------------------")

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def base_height_target(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reward maintaining a specific base height or reaching a height."""
    # Get the robot's root position
    root_pos_w = env.scene[asset_cfg.name].data.root_pos_w
    
    # Calculate error from target height (e.g., 1.0 meter)
    # We use a kernel (exp) to bound the reward between 0 and 1
    error = torch.square(root_pos_w[:, 2] - target_height)
    return torch.exp(-error / 0.25)

def jump_velocity_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reward vertical velocity (explosive jumping)."""
    root_vel_w = env.scene[asset_cfg.name].data.root_lin_vel_w
    # Only reward positive Z velocity (going up)
    vel_z = root_vel_w[:, 2]
    return torch.clamp(vel_z, min=0.0)

def feet_air_time(env, sensor_cfg: SceneEntityCfg, threshold: float):
    """Reward having feet in the air (to encourage lifting off)."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    # Check if feet are in contact (net forces > threshold)
    is_contact = torch.max(torch.norm(contact_sensor.data.net_forces_w, dim=-1), dim=1)[0] > threshold
    # Reward if NOT in contact
    return ~is_contact.float()
