import os
import random
import argparse
import glob

def parser():
    parser = argparse.ArgumentParser(description="split data to train.txt val.txt")
    parser.add_argument('-d', "--data_path", type=str,
                        help="please give the image directory")
    return parser.parse_args()

def check_arguments_errors(args):
    if not os.path.exists(args.data_path):
        raise(ValueError("Invalid data_path {}".format(os.path.abspath(args.data_path))))

def main():
    args = parser()
    check_arguments_errors(args)
    files = [filepath for filepath in glob.iglob(os.path.join(args.data_path, '*.png'))]
    random.shuffle(files)
    cut = int(len(files)*0.9)
    arr1 = files[:cut]
    arr2 = files[cut:]

    with open('train.txt', 'a+') as t:
        with open('val.txt', 'a+') as v:
            for i in arr1:
                t.writelines(i + '\n')
            for i in arr2:
                v.writelines(i + '\n')

if __name__ == '__main__':
    main()