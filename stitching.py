import os
import subprocess
import argparse
import shutil
import numpy as np
import cv2
import sys




def run_match_pairs_script(folder,relations_path):
    command = [
        "python", "match_pairs.py", "--resize", "-1", "--superglue", "outdoor",
        "--max_keypoints", "2048", "--nms_radius", "5", "--resize_float",
        "--input_dir", f"{folder}/tmp/", "--input_pairs", f"{relations_path}",
        "--output_dir", f"{folder}/output", "--keypoint_threshold", "0.05",
        "--match_threshold", "0.9"
    ]
    subprocess.run(command, capture_output=False)
    print("Match pairs script executed successfully")


def loadNPZ(npz_file, folder):
    npz = np.load(f'{folder}/output/'+ npz_file)
    point_set1 = npz['keypoints0'][npz['matches']>-1] # -1 if the keypoint is unmatched
    matching_indexes =  npz['matches'][npz['matches']>-1] 
    point_set2 = npz['keypoints1'][matching_indexes]
    print("Number of matching points for the findHomography algorithm:")
    print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))
    return point_set1, point_set2


def warpImages(im_right, im_left, H):

    print(type(im_left), type(im_right), type(H))

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
    corners_right_transformed = cv2.perspectiveTransform(np.float32([corners_right]), H)[0]
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
    H_t = offset @ H

    # Warp the right image
    panorama = cv2.warpPerspective(im_right, H_t, (w_panorama, h_panorama), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    panorama_right = panorama.copy()

    # Place the left image in the panorama
    panorama[-y_min:h_left - y_min, -x_min:w_left - x_min] = im_left

    panorama_left = np.zeros((h_panorama, w_panorama, 3), dtype=np.uint8)
    panorama_left[-y_min:h_left - y_min, -x_min:w_left - x_min]  = im_left


    print(-y_min,h_left, -x_min,w_left)

    return panorama, [w_left - x_min,  w_left - x_min], panorama_left, panorama_right



def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window)
    try:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])


def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side, showstep=False):
    """Given two aligned images @dst_img and @src_img_warped, and the @width_dst is width of dst_img
    before resize, that indicates where there is the discontinuity between the images,
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one"""

    h, w, _ = dst_img_rz.shape
    print(width_dst)
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window )
    print("barrier")
    print(barrier, width_dst, smoothing_window)
    print(h,w)
    mask1 = blendingMask(
        h, w, barrier , smoothing_window=smoothing_window, left_biased=False
    )
    mask2 = blendingMask(
        h, w, barrier , smoothing_window=smoothing_window, left_biased=True
    )

    if showstep:
        nonblend = src_img_warped + dst_img_rz
    else:
        nonblend = None
        leftside = None
        rightside = None

    if side == "left":
        #dst_img_rz = cv2.flip(dst_img_rz, 1)
        #src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped  = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz

        # normalize the panorama
        #pano = src_img_warped#cv2.flip(pano, 1)
        if showstep:
            leftside = cv2.flip(src_img_warped, 1)
            rightside = cv2.flip(dst_img_rz, 1)
    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        if showstep:
            leftside = dst_img_rz
            rightside = src_img_warped

    return pano, nonblend, leftside, rightside


def main(args):
    output_file = 'image_names.txt'

    os.makedirs(f"{args.folder}/tmp", exist_ok=True)

    images = sorted([f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))])

    for i in range(len(images)-1):
        print(i)
        if i == 0:
            shutil.copy2(f"{args.folder}/{images[i]}", f"{args.folder}/tmp/{images[i]}")
        else:
            shutil.copy2(path_panorama, f"{args.folder}/tmp/{path_panorama.split('/')[-1]}")
        shutil.copy2(f"{args.folder}/{images[i+1]}", f"{args.folder}/tmp/{images[i+1]}")

        if i == 0:
            with open(f"image_names.txt", 'w') as f:
                f.write(f"{images[i+1]} {images[i]}\n")
        else:
            with open(f"image_names.txt", 'w') as f:
                f.write(f"{images[i+1]} {path_panorama.split('/')[-1]}\n")

        run_match_pairs_script(args.folder ,output_file)


        if i == 0:
            os.remove(f"{args.folder}/tmp/{images[i]}")
        else:
            os.remove(f"{args.folder}/tmp/{path_panorama.split('/')[-1]}")
        os.remove(f"{args.folder}/tmp/{images[i+1]}")

        if i == 0:
            npz_path = f"{os.path.splitext(images[i+1])[0]}_{os.path.splitext(images[i])[0]}_matches.npz"
        else:
            npz_path = f"{os.path.splitext(images[i+1])[0]}_{os.path.splitext(path_panorama.split('/')[-1])[0]}_matches.npz"
        point_set1, point_set2 = loadNPZ(npz_path, args.folder)

        H, status = cv2.findHomography(point_set1, point_set2)

        if i == 0:
            im_right = cv2.imread(f"{args.folder}/{images[i+1]}")
        else:
            im_left = cv2.imread(f"{args.folder}/{images[i+1]}")
            
        if i == 0:
            im_left = cv2.imread(f"{args.folder}/{images[i]}")
        else:
            im_right = cv2.imread(path_panorama)
 

        if i == 0:
            panorama,t,panorama_left,panorama_right = warpImages(im_right, im_left, H)
        else:
            panorama,t,panorama_left,panorama_right  = warpImages(im_left, im_right, H)



        #blending panorama
        print(t)
        pano, nonblend, leftside, _ = panoramaBlending(
            panorama_right, panorama_left, t[0], side="left", showstep=False
        )

        #verify the size of the uimage and if its greater than 2000x2000 resize mantaining the aspect ratio
        if pano.shape[0] > 2000 or pano.shape[1] > 2000:
            if pano.shape[0] > pano.shape[1]:
                scale_factor = 1900/pano.shape[0]
            else:
                scale_factor = 1900/pano.shape[1]

            pano = cv2.resize(pano, (int(pano.shape[0]*scale_factor), int(pano.shape[1]*scale_factor)))


        path_panorama = f"{args.folder}/panorama_{images[i+1].split('.')[0]}_{images[i].split('.')[0]}.jpg"
        cv2.imwrite(path_panorama, pano)

        if i==1:
            break
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch images')
    parser.add_argument('--folder', type=str, help='Folder path containing images', default='./acquisitions')


    args = parser.parse_args()
    main(args)
