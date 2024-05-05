
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.objects import DynamicCapsule
import torch

class WoodView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "WoodView",
    ) -> None:
        """[summary]
        """
        print('===========',prim_paths_expr)
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=True
        )

        # grasp_pos_l
        # grasp_pos_r

        #self._grasp_pos_l = RigidPrimView(prim_paths_expr='/World/envs/.*/wood/base_link/grasp_pos_l',
        #                                  name='grasp_pos_l',
        #                                  reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        '''
        self._grasp_pos_l = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/ball",
                                           name="wood_grasp_l",
                                           position=torch.tensor[0,1,0])
        '''

        self._wood_base_links = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/base_link", 
                                                name="wood_view",
                                                reset_xform_properties=False,
                                                )

