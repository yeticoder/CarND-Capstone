import argparse
import cv2
import numpy as np
import copy
import os
import yaml

def change_brightness(img, br):
    # br = randint(0, 200) - 100
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], np.array([float(br)]))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def load_file(annotation, base_dir):
    filename = os.path.join(base_dir, annotation['filename'])

    return cv2.imread(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augmentation generator")
    parser.add_argument("--annotation_file", "-a", help="YAML annotations file", required=True)
    # parser.add_argument("--output_file", "-o", help="Output file", required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.annotation_file):
        raise IOError("Annotations file not found")

    with open(args.annotation_file, 'r') as f:
        annotations = yaml.load(f)

    base_dir = os.path.dirname(args.annotation_file)

    augmented_dir = os.path.join(base_dir, "aug")
    if not os.path.isdir(augmented_dir):
        os.makedirs(augmented_dir)

    output_yaml = os.path.join(base_dir, "augmented.yaml")

    annotations_aug = []

    for a in annotations:
        annotations_aug.append(a)

        basename = os.path.basename(a['filename'])
        filename, ext = os.path.splitext(basename)
        for br in [-50, 50]:
            aug_filename = "%s_%03d%s" % (filename, br, ext)
            aa = copy.deepcopy(a)
            aa['filename'] = os.path.join("aug", aug_filename)
            annotations_aug.append(aa)
            out_path = os.path.join(augmented_dir, aug_filename)

            img = load_file(a, base_dir)
            img_aug = change_brightness(img, br)

            cv2.imwrite(out_path, img_aug)


    if annotations_aug:
        with open(output_yaml, 'w') as f:
            yaml.dump(annotations_aug, f, default_flow_style=False)


