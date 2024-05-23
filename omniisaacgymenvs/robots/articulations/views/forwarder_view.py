
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class ForwarderView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ForwarderView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._grapple_body = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/grappleBody_endpoint", name="grappleBody_view", reset_xform_properties=False)
        self.unloading_point = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/unloading_point", name="unloading_point_view", reset_xform_properties=False)
        self.targets = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/target", name="target_view", reset_xform_properties=False)
        self._grapple_l = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/l_endpoint", name="grappleL_view", reset_xform_properties=False)
        self._grapple_r = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/r_endpoint",  name="grappleR_view", reset_xform_properties=False)
        self._base_link = RigidPrimView(prim_paths_expr="/World/envs/.*/forwarder/base_link", name="base_link_view", reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        
        #self._gripper_indices = [self.get_dof_index("grappleL1"), self.get_dof_index("grappleR1")]
        
    @property
    def gripper_indices(self):
        return self._gripper_indices
    
