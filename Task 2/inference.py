from src.matchers import SatelliteMatcher, BaseMatcher
import argparse
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--first-image', type=str,
                        help='Path to the first image.')
    parser.add_argument('--second-image', type=str,
                        help='Path to the second image.')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='A minimal score to consider match eligible.')
    parser.add_argument('--weights-point', type=str, default='weights/superpoint_v1.pth',
                        help='Path to the weights for the SuperPoint.')
    parser.add_argument('--weights-glue', type=str, default='weights/superglue_outdoor.pth',
                        help='Path to the weights for the SuperGlue.')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                        help='Scale the images.')
    parser.add_argument('-t', '--type', type=str, default='base',
                        help='Tells which matcher to use: "satellite" or "base".')

    return parser.parse_args()


def draw_match(image, pt1, pt2):
    pt1 = [int(v) for v in pt1]
    pt2 = [int(v) for v in pt2]

    offset = image.shape[1] // 2
    image = cv2.circle(image, pt1, radius=4, color=(0, 0, 255))
    image = cv2.circle(image, (pt2[0] + offset, pt2[1]), radius=4, color=(0, 0, 255))
    image = cv2.line(image, pt1, (pt2[0] + offset, pt2[1]), thickness=2, color=(0, 0, 255))

    return image


def inference_satellite(args):
    matcher = SatelliteMatcher()

    image_1 = cv2.imread(args.first_image)
    image_1 = cv2.resize(image_1, dsize=(8000, 8000))
    image_2 = cv2.imread(args.second_image)
    image_2 = cv2.resize(image_2, dsize=(8000, 8000))

    print(matcher(image_1=image_1, image_2=image_2))


def inference_base(args):
    intermidiate_size = 480

    matcher = BaseMatcher(
        {
            'preprocessor': {
                'output_size': intermidiate_size
                },
            'super_point_path': args.weights_point,
            'super_glue_path': args.weights_glue
        })

    # Make image square
    image_1 = cv2.imread(args.first_image)
    image_2 = cv2.imread(args.second_image)

    # Find the biggest height out of images. It will be our size.
    size = max(image_1.shape[0], image_2.shape[1])

    image_1 = cv2.resize(image_1, dsize=(size, size))
    image_2 = cv2.resize(image_2, dsize=(size, size))

    # Perform an inference
    result = matcher(image_1=image_1, image_2=image_2)

    image_1 = cv2.resize(image_1, dsize=[int(size * args.scale), int(size * args.scale)])
    image_2 = cv2.resize(image_2, dsize=[int(size * args.scale), int(size * args.scale)])

    keypoints_1, keypoints_2 = result['keypoints0'][0], result['keypoints1'][0]
    matches, scores = result['matches0'][0], result['matching_scores0'][0]
    
    # Throw away matches with low threshold
    for index, score in enumerate(scores):
        if score < args.threshold:
            matches[index] = -1

    mask = matches != -1
    points_1 = keypoints_1[mask] * size / intermidiate_size * args.scale
    points_2 = keypoints_2[matches[mask]] * size / intermidiate_size * args.scale

    # Draw matched on the image and save it.
    out_image = cv2.hconcat([image_1, image_2], None)
    for index, keypoint in enumerate(points_1):
        out_image = draw_match(out_image, keypoint, points_2[index])
    
    cv2.imwrite('matched_output.jpg', out_image)


if __name__ == '__main__':
    args = parse_arguments()

    if args.type == 'satellite':
        inference_satellite(args=args)
    elif args.type == 'base':
        inference_base(args=args)


    print('Inferencing has done!')