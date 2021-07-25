from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .messytable_dataset import MessytableDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "messytable": MessytableDataset
}
