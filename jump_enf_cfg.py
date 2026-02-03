from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg

# --- 1. Import the ROUGH environment parent class ---
from .rough_env_cfg import H1RoughEnvCfg 

# --- 2. Import Terrain Configs ---
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains import HfPyramidSlopedTerrainCfg
from isaaclab.terrains import HfDiscreteObstaclesTerrainCfg
from isaaclab.terrains import MeshPlaneTerrainCfg

# Import your custom rewards
from . import jump_rewards

class H1JumpRoughEnvCfg(H1RoughEnvCfg):
    """
    Jumping Task on ROUGH Terrain for Unitree H1.
    """
    def __post_init__(self):
        # 1. Load standard H1 Rough config
        super().__post_init__()

        # =====================================================
        # A. CONFIGURE TERRAIN
        # =====================================================
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.5, 1.0), 
            sub_terrains={
                "pyramid_slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.4, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "boxes": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.4, 
                    num_obstacles=20, 
                    obstacle_height_range=(0.1, 0.5), 
                    obstacle_width_range=(0.5, 1.0),
                    platform_width=2.0
                ),
                "flat": MeshPlaneTerrainCfg(proportion=0.2),
            },
        )

        # =====================================================
        # B. SYNTHETIC DATA / DOMAIN RANDOMIZATION
        # =====================================================
        
        # --- REMOVED FRICTION RANDOMIZATION TO FIX CRASH ---
        # (The function signature for material randomization varies between versions.
        # We skip it to ensure the terrain geometry loads correctly.)
        
        # Randomize Mass (Robot Payload)
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add",
            },
        )

        # Random Pushes (Wind/Shoves)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # =====================================================
        # C. REWARDS
        # =====================================================
        # Disable Walking
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 0.0
        
        # Penalize bad orientation
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-10.0,
        )
        
        # Jumping Rewards
        self.rewards.target_jump_height = RewTerm(
            func=jump_rewards.base_height_target,
            weight=30.0,
            params={"target_height": 1.30, "asset_cfg": SceneEntityCfg("robot")}
        )

        self.rewards.upward_thrust = RewTerm(
            func=jump_rewards.jump_velocity_z,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # =====================================================
        # D. COMMANDS
        # =====================================================
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
