from ..processors import SatellitePreprocessor
from ..models import SuperPoint
import torch, cv2
import datetime, os, glob


class SatelliteKeypointInferencer():
    '''
            This class takes care of extensive amounts of memory by storing every inference in
        a temporary file 'inference_${TIME}.txt'.
    '''
    def __init__(self, config: dict={}):
        self.preprocessor = SatellitePreprocessor(config=config.get('preprocessor', {}))
        self.point_detector = SuperPoint(config=config.get('point_detector', {}))
        weights_path = config.get('weights', 'weights/superpoint_v1.pth')
        self.point_detector.load_state_dict(torch.load(weights_path))
        self.batch_size = config.get('batch_size', -1) # Currently unsupported, but can increase speed.
        self.device = config.get('device', 'cpu')

        if self.device == 'cuda':
            self.point_detector = self.point_detector.to(self.device)


    def inference(self, image: cv2.Mat) -> str:
        '''
            Iterates over child images and generates keypoints for each of them.
        Each result of inference is being written to the temporary file in json format.

            returns name of the file, where all information is being stored.
        '''
        storage_folder_name = 'temp_inference_storage_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        os.mkdir(storage_folder_name)
        child_images = self.preprocessor(image=image)

        for index, image in enumerate(child_images):
            inference_result = self.point_detector(
                data = {'image': torch.unsqueeze(image, dim=0).to(self.device)}
            )
            torch.save(inference_result, os.path.join(storage_folder_name, '%04d' % index + '.pt'))

        del child_images

        return storage_folder_name

    
    def draw_keypoints(self, image: cv2.Mat, keypoints_folder: str) -> cv2.Mat:
        '''
            Function inferences the satellite image and finds keypoints. In the result it
        gets an image.

            Warning: currently forcibly resizes image to (8160, 8160)! 
        '''
        inferences_names = sorted(glob.glob('*.pt', root_dir=keypoints_folder))

        row_num = col_num = image.shape[0] // self.preprocessor.input_size
        size = self.preprocessor.input_size

        for row in range(row_num):
            for column in range(col_num):
                path_index = row * row_num + column
                filename = os.path.join(keypoints_folder, inferences_names[path_index])
                print('Currently draw section (%d, %d), filename: %s' % (row, column, filename))
                inference_result = torch.load(filename, map_location=lambda storage, loc: storage)
                keypoints = inference_result['keypoints'][0]
                for keypoint in keypoints:
                    pt = [keypoint[0] + size * column, keypoint[1] + size * row]
                    pt = [int(v) for v in pt]
                    image = cv2.circle(image, pt, radius=3, color=(0, 0, 255))

        return image