from ..models import SuperGlue, SuperPoint
from ..processors import BasePreprocessor
import torch


class BaseMatcher():
    '''
            Class for matching regular images using SuperPoint and SuperGlue.
    '''

    def __init__(self, config: dict={}):
        self.preprocessor = BasePreprocessor(config=config.get('preprocessor', {}))

        self.point_detector = SuperPoint(config.get('super_point', {}))
        point_detector_path = config.get('super_point_path', 'weights/superpoint_v1.pth')
        self.point_detector.load_state_dict(torch.load(point_detector_path))

        self.point_matcher = SuperGlue(config=config.get('super_glue', {}))
        point_matcher_path = config.get('super_glue_path', 'weights/superglue_outdoor.pth')
        self.point_matcher.load_state_dict(torch.load(point_matcher_path))


    def __call__(self, image_1, image_2):
        image_1 = self.preprocessor(image_1)
        image_2 = self.preprocessor(image_2)
        extracted_data_1 = self.point_detector(data={'image':image_1})
        extracted_data_2 = self.point_detector(data={'image':image_2})

        pred = {key + '0': value for key, value in extracted_data_1.items()}
        pred = {**pred, **{key + '1': value for key, value in extracted_data_2.items()}}

        data = {
            **pred,
            'image0': image_1,
            'image1': image_2
        }

        for key in data.keys():
            if isinstance(data[key], (list, tuple)):
                data[key] = torch.stack(data[key])

        result = self.point_matcher(data=data)

        return {**result, **data}