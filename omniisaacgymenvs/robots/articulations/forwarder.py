from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema
import pathlib

class Forwarder(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "forwarder",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name
        self._position = torch.tensor([2.0, 2.0, 2.5]) if translation is None else translation
        self._orientation = torch.tensor([0, 0, 0, 1]) if orientation is None else orientation
        self._usd_path = str(pathlib.Path(__file__).parent.resolve().parent/'fwd_assets/forwarder.usd')

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "base_link/Revolute_1",
            "cranepillar/Revolute_2",
            "cranearm/Revolute_3",
            "extension_arm/Slider_1",
            "extension/Revolute_4",
            "intermediatehookjnt/Revolute_5",
            "grapplehook/Revolute_6",
            "grapplebody/Revolute_7",
            "grapplebody/Revolute_8"
        ]

        drive_type = ['angular',
                      'angular',
                      'angular',
                      'linear',
                      'free',
                      'free',
                      'angular',
                      'angular',
                      'angular'
                      ]
        
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.1, 0.1]]
        stiffness =    [1e8,1e8, 1e8, 1e8,          0, 0       ,1e8, 1e8, 1e8]
        damping =      [100000.0,  100000.0,   100000.0,   100000.0,          0, 0          ,100000.0,  100000.0,   100000.0]
        max_velocity = [15,15,15,2,                 500,500,    50,50,50]
        max_force  =   [8e5,5e5,5e5,5e3,              0,0,   5e5, 5e6,5e6]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])

        