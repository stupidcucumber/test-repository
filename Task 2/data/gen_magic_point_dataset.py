import pandas as pd
import tarfile, glob, os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', type=str, default='data/MagicPoint',
                        help='Path to the folder containing dataset files.')
    parser.add_argument('-o', '--output', type=str, default='magic_point_dataset.csv',
                        help='Name of the file, that stores dataset.')
    parser.add_argument('--extract-tar', action='store_true',
                        help='Whether to extract .tar archives, or they have been already extracted. Default to False.')
    
    return parser.parse_args()


def extract_tar_names(folder: str) -> list:
    names = glob.glob('*.tar*', root_dir=folder)
    return [os.path.join(folder, name) for name in names]


def extract_folder_names(folder: str) -> list:
    names = glob.glob('*', root_dir=folder)
    paths = [os.path.join(folder, name) for name in names]
    return [path for path in paths if os.path.isdir(path)]


def extract_file_names(folder: str, pattern: str) -> list:
    names = glob.glob(pattern, root_dir=folder)
    return [os.path.join(folder, name) for name in names]


def unpack_tar_files(folder_in: str):
    paths = extract_tar_names(folder=folder_in)

    for path in paths:
        file = tarfile.open(path)
        file.extractall(path=folder_in)


def build_pandas_dataframe(root_folder: str) -> pd.DataFrame:
    result = [] # pd.DataFrame(columns=['image_path', 'label_path', 'partition'])
    partitions = ['test', 'training', 'validation']
    image_paths = []

    folder_paths = extract_folder_names(root_folder)

    for partition in partitions:
        temp = []
        for path in folder_paths:
            image_folder_path = os.path.join(path, 'images')
            partition_image_paths = extract_file_names(os.path.join(image_folder_path, partition), '*.png')
            temp.extend(partition_image_paths)
        image_paths.append(temp)

    for index, partition in enumerate(partitions):
        for image_path in image_paths[index]:
            result.append(
                {
                    'image_path': image_path,
                    'label_path': image_path.replace('images', 'points').replace('png', 'npy'),
                    'partition': partition
                }
            )

    return pd.DataFrame(result)


if __name__ == '__main__':
    args = parse_arguments()

    if args.extract_tar:
        unpack_tar_files(args.input_folder)

    dataframe = build_pandas_dataframe(args.input_folder)

    dataframe.to_csv(args.output, index=False)

