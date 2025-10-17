
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
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=True
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._wood_base_links = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/base_link", 
                                                name="wood_view",
                                                reset_xform_properties=False,
                                                )

