from ..models import SuperGlue, SuperPoint
from ..processors import SatellitePreprocessor
import torch


class SatelliteMatcher():
    def __init__(self, config: dict = {}):
        self.processor = SatellitePreprocessor()

        self.point_detector = SuperPoint(config=config.get('super_point', {}))
        point_detector_weights_path = config.get('super_point_wp', 'weights/superpoint_v1.pth')
        self.point_detector.load_state_dict(torch.load(point_detector_weights_path))

        self.point_matcher = SuperGlue(config=config.get('super_glue', {}))
        point_matcher_weights_path = config.get('super_glue_wp', 'weights/superglue_outdoor.pth')
        self.point_matcher.load_state_dict(torch.load(point_matcher_weights_path))


    def __call__(self, image_1, image_2):
        '''
                So we split up image_1 from satelite into smaller ones. The same we do to the image_2. 
            After running them through SuperPoint: we get a bunch of images and their 
            corresponding keypoints and descriptions.
                All those keypoints we pass through the SuperGlue, where model matches all the available
            keypoints. 
        '''
        images_1 = self.processor(image=image_1)
        images_2 = self.processor(image=image_2)

        extracted_data_1 = self.point_detector({'image': images_1})
        extracted_data_2 = self.point_detector({'image': images_2})
        
        