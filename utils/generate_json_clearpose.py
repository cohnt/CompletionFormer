"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    This script generates a json file for the ClearPose dataset.
    https://github.com/opipari/ClearPose
"""

import os
import argparse
import random
import json
import numpy as np
import scipy.io

parser = argparse.ArgumentParser(
    description="ClearPose Depth Completion json generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the ClearPose dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='clearpose.json', help="Output file name")
parser.add_argument('--val_ratio', type=float, required=False,
                    default=0.05, help='Validation data ratio')
parser.add_argument('--test_ratio', type=float, required=False,
                    default=0.20, help='Validation data ratio')
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e10), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')

args = parser.parse_args()

assert float(args.val_ratio) + float(args.test_ratio) < 1.0

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def generate_json():
    os.makedirs(args.path_out, exist_ok=True)
    check_dir_existence(args.path_out)

    list_data = []
    sets = ["set%d" % i for i in range(1, 9+1)]
    for s in sets:
        set_path = args.path_root + "/" + s
        scenes = os.listdir(set_path)
        for scene in scenes:
            if s == "set6" and scene == "scene1":
                # Missing metadata, so no intrinsics
                continue
            if s == "set8" and scene == "scene1":
                # Missing metadata, so no intrinsics
                continue

            scene_path = set_path + "/" + scene
            files = os.listdir(scene_path)
            print(s, scene)
            # Get the actual numbers while filtering out "metadata.mat" and files of the form "CL-###...".
            nums = [int(file.split("-")[0]) for file in files if file[-3:] == "png" and file[:2] != "CL"]

            metadata_path = scene_path + "/metadata.mat"
            metadata = scipy.io.loadmat(metadata_path)

            for num in np.unique(nums):
                string_num = str(num).zfill(6)

                path_rgb = scene_path + "/" + string_num + "-color.png"
                path_depth = scene_path + "/" + string_num + "-depth.png"
                path_gt = scene_path + "/" + string_num + "-depth_true.png"
                path_calib = scene_path + "/" + string_num + "-calib.txt"

                # If the calibration file doesn't exist already, create it
                # by reading the data from metadata.mat

                if not os.path.isfile(path_calib):
                    K = metadata[string_num][0,0][3]
                    assert K.shape == (3, 3)
                    assert K[1,0] == 0
                    assert K[0,1] == 0
                    np.savetxt(path_calib, K)

                dict_sample = {
                    'rgb': path_rgb,
                    'depth': path_depth,
                    'gt': path_gt,
                    'K': path_calib
                }
                list_data.append(dict_sample)

    print("Loaded %d images" % len(list_data))

    random.shuffle(list_data)
    test_start = 0
    test_stop = test_start + int(args.test_ratio * len(list_data))
    val_start = test_stop
    val_stop = val_start + int(args.val_ratio * len(list_data))
    train_start = val_stop
    train_stop = len(list_data)

    if test_stop - test_start > args.num_test:
        test_stop = test_start + args.num_test
    if val_stop - val_start > args.num_val:
        val_stop = val_start + args.num_val
    if train_stop - train_start > args.num_train:
        train_stop = train_start + args.num_train

    dict_json = {}
    dict_json["train"] = list_data[train_start:train_stop]
    dict_json["val"] = list_data[val_start:val_stop]
    dict_json["test"] = list_data[test_start:test_stop]

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    generate_json()
