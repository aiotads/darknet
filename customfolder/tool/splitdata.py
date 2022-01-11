import os
import random
import argparse
import glob
from pathlib import Path

def parser():
    parser = argparse.ArgumentParser(description="split data to train.txt val.txt")
    parser.add_argument('-ip', "--image_path", type=str,
                        help="please give the image directory")
    parser.add_argument('-r', "--ratio", type=float, default=0.9,
                        help="please give the image directory")
    return parser.parse_args()

def check_arguments_errors(args):
    if not os.path.exists(args.image_path):
        raise(ValueError("Invalid image_path {}".format(os.path.abspath(args.image_path))))

def main():
    args = parser()
    check_arguments_errors(args)

    files = [str(path) for path in Path(args.image_path).rglob('*.png')]
    random.shuffle(files)
    cut = int(len(files)*args.ratio)
    arr1 = files[:cut]
    arr2 = files[cut:]

    with open('train.txt', 'w+') as train, open('val.txt', 'w+') as val:
        train.writelines('\n'.join(s for s in arr1))
        val.writelines('\n'.join(s for s in arr2))
    
if __name__ == '__main__':
    main()