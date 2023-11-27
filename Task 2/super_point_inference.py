from src.models import SuperPoint, SatelliteKeypointInferencer
from src.processors import BasePreprocessor
import argparse
import torch, cv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weight-path', type=str, default='weights/superpoint_v1.pth',
                        help='Path to the pretrained SuperPoint model.')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the image to inference on.')
    parser.add_argument('--intermediate-size', type=int, default=480,
                        help='Size of the image that is being sended to the input of the model.')
    parser.add_argument('-o', '--output', type=str, default='output.jpg',
                        help='Name of the output file (file that contain predicitons.)')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='On which device to inference.')
    parser.add_argument('--from-folder', type=str, default='',
                        help='If model had already inferenced and you only need to draw.')
    parser.add_argument('-t', '--type', type=str, default='base',
                        help='Type of the inference (whether to use temp-backed inference.)')
    
    return parser.parse_args()


def draw_keypoints(image, keypoints, xsize_scaler, ysize_scaler):
    for point in keypoints:
        image = cv2.circle(image, 
                           center=[int(point[0] * xsize_scaler), int(point[1] * ysize_scaler)], 
                           radius=2, 
                           color=(0, 0, 255))

    return image


def inference_satellite(args) -> cv2.Mat:
    inferencer = SatelliteKeypointInferencer()

    image = cv2.imread(args.input)
    image = cv2.resize(image, dsize=(8160, 8160))
    if args.from_folder == '':
        folder_name = inferencer.inference(image)
    else:
        folder_name = args.from_folder

    output_image = inferencer.draw_keypoints(image=image, keypoints_folder=folder_name)
    return output_image


def inference_base(args) -> cv2.Mat:
    '''
        Function for inference image under "base" workflow. Do not cuts image in parts.
    '''
    model = SuperPoint()
    model.load_state_dict(torch.load(args.weight_path))
    model = model.to(args.device)
    preprocessor = BasePreprocessor({'input_size': args.intermediate_size})

    image = cv2.imread(args.input)
    height, width, _ = image.shape

    preprocessed_image = preprocessor(image).to(args.device)
    result = model({
        'image': preprocessed_image
    })

    output_image = draw_keypoints(image,
                                  result['keypoints'][0],
                                  xsize_scaler=width / args.intermediate_size,
                                  ysize_scaler=height / args.intermediate_size)
    
    return output_image


if __name__ == '__main__':
    args = parse_arguments()

    if args.type == 'base':
        output_image = inference_base(args)
    elif args.type == 'satellite':
        output_image = inference_satellite(args)
    else:
        raise ValueError('No such type of inference! Choose between "base" and "satellite".')

    cv2.imwrite(args.output, output_image)
    print('Inference is over! Saving as %s!' % args.output)