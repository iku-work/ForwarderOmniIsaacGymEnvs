from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pathlib import Path
import numpy as np
import torch

class Wood(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "wood",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """
        self._usd_path = usd_path
        self._name = name
        self._usd_path = str(Path(__file__).parent.resolve().parent/'fwd_assets/wood.usd')

        add_reference_to_stage(self._usd_path, prim_path)

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.1, 0.0, 0.0, 0.0]) if orientation is None else orientation

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
