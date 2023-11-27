from ..models import SuperGlue, SatelliteKeypointInferencer
import torch


class SatelliteMatcher():
    def __init__(self, config: dict = {}):
        self.point_detector = SatelliteKeypointInferencer(
            config={
                'weights': config.get('point_detector', 'weights/superpoint_v1.pth'),
                'device': config.get('device', 'cpu')
            }
        )

        self.point_matcher = SuperGlue(config=config.get('super_glue', {}))
        point_matcher_weights_path = config.get('super_glue_wp', 'weights/superglue_outdoor.pth')
        self.point_matcher.load_state_dict(torch.load(point_matcher_weights_path))
        self.point_matcher = self.point_matcher.to(config.get('device', 'cpu'))


    def _load_inference_folder(folder_path: str, data: dict={}):
        pass


    def __call__(self, image_1, image_2):
        '''
                So we split up image_1 from satelite into smaller ones. The same we do to the image_2. 
            After running them through SuperPoint: we get a bunch of images and their 
            corresponding keypoints and descriptions.
                All those keypoints we pass through the SuperGlue, where model matches all the available
            keypoints.
        '''
        inferenced_folder_1 = self.point_detector.inference(image=image_1)
        inferenced_folder_2 = self.point_detector.inference(image=image_2)

        data = {}
        pass
        
        
        