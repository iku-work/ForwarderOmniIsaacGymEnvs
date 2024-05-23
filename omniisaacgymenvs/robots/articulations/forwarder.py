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
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        '''if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/Franka/franka_instanceable.usd"
        print('================= USD PATH', self._usd_path)'''



        self._usd_path =  '/home/rl/Documents/forwarder_description/forwarder.usd'
        print("========================",str(pathlib.Path(__file__).parent.resolve().parent/'fwd_assets/forwarder.usd'))
        self._usd_path = str(pathlib.Path(__file__).parent.resolve().parent/'fwd_assets/forwarder.usd')
        #(pathlib.Path(__file__).parent.resolve().parent).joinpath('fwd_assets/forwarder.usd')

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        

        '''dof_paths = [
            "base_link/Revolute_1",
            "cranePillar1/Revolute_25",
            "craneArm1/Revolute_36",
            "extension_arm1/Slider_37",
            "extension1/Revolute_38",
            "intermediateHookJnt1/Revolute_16",
            "grappleHook1/Revolute_26",
            "grappleBody1/Revolute_30",
            "grappleBody1/Revolute_31"
        ]'''

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
                      ] #['angular'] * len(dof_paths)
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.1, 0.1]]
        
        '''
        # Works with these parameters
        stiffness =    [1e8,1e8,1e8,1e8,           1e8,1e8,        1e8,5e8,5e8]
        damping =      [5e6,1e6,1e6,1e6,           1e8,1e8,        1e5,1e8,1e8]
        max_force =    [5e5, 5e5, 5e5, 5e3,        5e3,5e3,        5e5,5e6,5e6]
        max_velocity = [1000,1000,1000,1000,1000,  1000,1000,  1000,1000]
        
        stiffness =    [1e8,1e8,1e8,1e8,           1e8,1e8,        1e8,5e8,5e8]#5e8,5e9
        damping =      [5e6,1e6,1e6,1e6,           1e8,1e8,        1e5,1e8,1e8]#1e3,1e3]
        max_force =    [5e5, 5e5, 5e5, 5e3,        5e3,5e3,        5e5,5e7,5e7]#5e7,5e7]
        max_velocity = [500,500,500,500,500,  500,500,  1000,1000]
        '''
        stiffness =    [10000000.0,10000000.0, 10000000.0, 10000000.0,          0, 0       ,10000000.0, 100000000.0, 100000000.0]
        damping =      [100000.0,  100000.0,   100000.0,   100000.0,          0, 0          ,100000.0,  100000.0,   100000.0]
        
        # VALUE: 6500
        #max_velocity = [30,30,30,40,                 500,500,    200,200,200]
        # VALUE: 180748
        #max_velocity = [30,30,30,10,                 500,500,    200,200,200]
        max_velocity = [15,15,15,2,                 500,500,    50,50,50]
        #max_force  =  [50000.0,500000.0,80000.0,2000.0,                    0,0,           10000.0,10000.0,10000.0]
        max_force  =   [8e5,5e5,5e5,5e3,              0,0,   5e5, 5e6,5e6]
        #max_force  =   [5e5,5e6,2e6,5e5,            1e20,1e20,   1e5, 5e4,5e4]
        


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

        