import os
import subprocess
import argparse
import shutil
import numpy as np
import cv2


def create_image_pairs(folder_path, output_file):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpeg', '.jpg', '.png')])
    
    with open(output_file, 'w') as f:
        f.writelines(f"{files[i]} {files[i-1]}\n" for i in range(1, len(files)))

    npz_files = [f"{os.path.splitext(files[i])[0]}_{os.path.splitext(files[i-1])[0]}_matches.npz" for i in range(1, len(files))]

    return files, npz_files


def run_match_pairs_script(folder,relations_path):
    command = [
        "python3", "match_pairs.py", "--resize", "-1", "--superglue", "outdoor",
        "--max_keypoints", "2048", "--nms_radius", "5", "--resize_float",
        "--input_dir", f"{folder}/tmp/", "--input_pairs", f"{relations_path}",
        "--output_dir", f"{folder}/output", "--keypoint_threshold", "0.05",
        "--match_threshold", "0.9"
    ]
    subprocess.run(command)


def loadNPZ(npz_file, folder):
    npz = np.load(f'{folder}/output/'+ npz_file)
    point_set1 = npz['keypoints0'][npz['matches']>-1] # -1 if the keypoint is unmatched
    matching_indexes =  npz['matches'][npz['matches']>-1] 
    point_set2 = npz['keypoints1'][matching_indexes]
    print("Number of matching points for the findHomography algorithm:")
    print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))
    return point_set1, point_set2


def warpImages(im_left, im_right, H):

    # Calculate the size of the output panorama canvas
    h_left, w_left = im_left.shape[:2]
    h_right, w_right = im_right.shape[:2]

    # Corners of the right image
    corners_right = np.array([
        [0, 0],
        [w_right - 1, 0],
        [w_right - 1, h_right - 1],
        [0, h_right - 1]
    ])

    # Transform corners to get the bounding box of the warped right image
    corners_right_transformed = cv2.perspectiveTransform(np.float32([corners_right]), np.linalg.inv(H))[0]
    corners_all = np.vstack((corners_right_transformed, [[0, 0], [w_left, 0], [0, h_left], [w_left, h_left]]))

    # Find the extents of both the transformed and original images
    x_min, y_min = np.min(corners_all, axis=0).astype(int)
    x_max, y_max = np.max(corners_all, axis=0).astype(int)

    # Size of the panorama
    w_panorama = x_max - x_min
    h_panorama = y_max - y_min

    # Offset for translation
    offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Update the transformation matrix based on offset
    H_t = offset @ np.linalg.inv(H)

    # Warp the right image
    panorama = cv2.warpPerspective(im_right, H_t, (w_panorama, h_panorama), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Place the left image in the panorama
    panorama[-y_min:h_left - y_min, -x_min:w_left - x_min] = im_left

    return panorama


def main(args):
    output_file = 'image_names.txt'

    os.makedirs(f"{args.folder}/tmp", exist_ok=True)

    print("args.folder", args.folder)
    images = sorted([f for f in os.listdir(args.folder)])

    for i in range(len(images)):
        shutil.copy(f"{args.folder}/{images[i]}", f"{args.folder}/tmp/{images[i]}")
        shutil.copy(f"{args.folder}/{images[i+1]}", f"{args.folder}/tmp/{images[i+1]}")

        run_match_pairs_script(args.folder ,output_file)


        os.remove(f"{args.folder}/tmp/{images[i]}")
        os.remove(f"{args.folder}/tmp/{images[i+1]}")


        npz_path = f"{os.path.splitext(images[i+1])[0]}_{os.path.splitext(images[i])[0]}_matches.npz"
        point_set1, point_set2 = loadNPZ(npz_path, args.folder)
        H, status = cv2.findHomography(point_set1, point_set2, cv2.RANSAC, 5.0)

        im_left = cv2.imread(f"{args.folder}/{images[i]}")
        im_right = cv2.imread(f"{args.folder}/{images[i+1]}")

        panorama = warpImages(im_right, im_left, H)

        cv2.imwrite(f"{args.folder}/tmp/panorama_{images[i+1]}_{images[i]}.png", panorama)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch images')
    parser.add_argument('--folder', type=str, help='Folder path containing images', default='./acquisitions')


    args = parser.parse_args()
    main(args)
