from subprocess import check_output

from images.camvid import CamvidDataset  # noqa
from images.cityscapes import CityscapesDataset  # noqa
from images.isbi_em_stacks import IsbiEmStacksDataset  # noqa
from images.kitti import KITTIdataset  # noqa
from images.mscoco import MSCocoDataset  # noqa
from images.pascalvoc import PascalVOCdataset  # noqa
from images.polyps912 import Polyps912Dataset  # noqa
from images.scene_parsing_MIT import SceneParsingMITDataset  # noqa

from videos.change_detection import ChangeDetectionDataset  # noqa
from videos.davis import DavisDataset  # noqa
from videos.gatech import GatechDataset  # noqa

__version__ = check_output('git rev-parse HEAD',
                           shell=True).strip().decode('ascii')
