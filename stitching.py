import os
import subprocess
import argparse
import shutil
import numpy as np
import cv2
import sys
import re


def run_match_pairs_script(folder,relations_path):
    """
    Runs the match_pairs.py script with specified arguments, suppressing its output.

    Args:
        folder (str): The directory containing input files and where output will be stored.
        relations_path (str): The path to the file containing the relations data.

    Raises:
        subprocess.CalledProcessError: If the match_pairs.py script exits with a non-zero status.
    """

    command = [
        sys.executable, "match_pairs.py", "--resize", "-1", "--superglue", "outdoor",
        "--max_keypoints", "2048", "--nms_radius", "5", "--resize_float",
        "--input_dir", f"{folder}/tmp/", "--input_pairs", f"{relations_path}",
        "--output_dir", f"{folder}/output", "--keypoint_threshold", "0.05",
        "--match_threshold", "0.9"
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Match pairs script executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing match pairs script: {e}")


def loadNPZ(npz_file, folder):
    """
    Loads matching keypoints from an NPZ file.

    Args:
        npz_file (str): The name of the NPZ file to be loaded.
        folder (str): The directory containing the NPZ file.

    Returns:
        tuple: Two numpy arrays, point_set1 and point_set2, containing the matching keypoints
               from the left and right images respectively.

    """
    npz = np.load(f'{folder}/output/'+ npz_file)

    point_set1 = npz['keypoints0'][npz['matches']>-1] # -1 if the keypoint is unmatched
    matching_indexes =  npz['matches'][npz['matches']>-1] 
    point_set2 = npz['keypoints1'][matching_indexes]

    print("Number of matching points for the find Homography algorithm:")
    print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))

    return point_set1, point_set2


def warpImages(im_right, im_left, H):
    """
    Creates a panorama by stitching two images together using a given homography matrix.

    Args:
        im_left (numpy.ndarray): The left image.
        im_right (numpy.ndarray): The right image.
        H (numpy.ndarray): The homography matrix that warps the right image to align with the left image.

    Returns:
        tuple: A tuple containing:
            - panorama (numpy.ndarray): The resulting stitched panorama image.
            - offsets (list): The result and offset values used for translation.
            - panorama_left (numpy.ndarray): The panorama containing only the left image.
            - panorama_right (numpy.ndarray): The panorama containing only the warped right image.
            - corners_all (numpy.ndarray): The coordinates of the corners of both images.
    """

    # Calculate the size of the output panorama canvas
    h_left, w_left = im_left.shape[:2]
    h_right, w_right = im_right.shape[:2]

    # Corners of the left image
    corners_left = np.array([
        [0, 0], # top-left
        [w_right - 1, 0], # top-right
        [w_right - 1, h_right - 1], # bottom-right
        [0, h_right - 1] # bottom-left
    ])

    # Transform corners to get the bounding box of the warped right image
    corners_right_transformed = cv2.perspectiveTransform(np.float32([corners_left]), H)[0]

    if corners_right_transformed[1][0] < corners_right_transformed[2][0]:
        corner_overlap = corners_right_transformed[1][0] # top-right
    else:
        corner_overlap = corners_right_transformed[2][0] # bottom-right

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

    panorama[-y_min:h_left - y_min, -x_min:w_left - x_min] = im_left

    panorama_left = np.zeros((h_panorama, w_panorama, 3), dtype=np.uint8)
    panorama_left[-y_min:h_left - y_min, -x_min:w_left - x_min]  = im_left

    corner_overlap = int(corner_overlap - x_min)
    offset = corner_overlap - (-x_min)

    return panorama, [corner_overlap,  offset], panorama_left, panorama_right, corners_all


def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    """
    Creates a blending mask for image stitching with a smooth transition at the barrier.

    Args:
        height (int): The height of the mask.
        width (int): The width of the mask.
        barrier (int): The column index where the transition occurs.
        smoothing_window (int): The number of columns over which the transition from 1 to 0 or 0 to 1 is smoothed.
        left_biased (bool, optional): If True, the mask is 1 on the left side of the barrier and transitions to 0 on the right.
                                      If False, the mask is 0 on the left side of the barrier and transitions to 1 on the right.

    Returns:
        numpy.ndarray: A 3-channel blending mask with smooth transitions for image stitching.
    """
    assert barrier < width, "Barrier index must be less than the width of the mask."
    
    mask = np.zeros((height, width))
    offset = int(smoothing_window)

    if left_biased:

        mask[:, barrier - offset : barrier + 1] = np.tile(
            np.linspace(1, 0, offset + 1).T, (height, 1)
        )
        mask[:, : barrier - offset] = 1
    else:

        mask[:, barrier - offset : barrier  + 1 ] = np.tile(
            np.linspace(0, 1,  offset + 1).T, (height, 1)
        )
        mask[:, barrier :] = 1

    return cv2.merge([mask, mask, mask])


def panoramaBlending(dst_img_rz, src_img_warped, width_dst,smoothing_window):
    """
    Blends two aligned images with a smooth transition in the overlapping area.

    Args:
        dst_img_rz (numpy.ndarray): The destination (left) image resized.
        src_img_warped (numpy.ndarray): The source (right) image warped to align with the destination image.
        width_dst (int): The original width of the destination image before resizing.
        smoothing_window (int): The width of the transient smoothing window.

    Returns:
        pano (numpy.ndarray): The resulting blended panorama image.
    """

    h, w, _ = dst_img_rz.shape 

    mask1 = blendingMask(
        h, w, width_dst , smoothing_window=smoothing_window, left_biased=False
    )
    mask2 = blendingMask(
        h, w, width_dst , smoothing_window=smoothing_window, left_biased=True
    )

    # element-wise multiplication
    dst_img_rz = dst_img_rz * mask2
    cv2.imwrite("mask1.png", mask1 * 255)
    src_img_warped  = src_img_warped * mask1
    cv2.imwrite("mask2.png", mask2 * 255)

 
    # add two images
    pano = src_img_warped + dst_img_rz

    return pano


def crop(panorama, h_dst, conners):
    """crop panorama based on destination.
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and
    4 conners of destination image"""

    xmin , ymindest =  np.min(conners, axis=0).astype(int)

    # Determine the ymin
    if conners[0][1] < conners[1][1]:
        if conners[1][1] < ymindest:
            ymin = ymindest
        else:
            
            ymin = conners[1][1]
            if ymindest <  -conners[1][1]:
                ymin = np.abs(ymindest) + conners[1][1]
    else:
        if conners[0][1] < ymindest:
            ymin = ymindest
        else:
            ymin = conners[0][1]

    # Determine the ymax
    if conners[2][1] > conners[3][1]:
        if conners[3][1] > ymindest + h_dst:
            ymax = ymindest + h_dst
        else:
            ymax = conners[3][1]
    else:
        if conners[2][1] > ymindest + h_dst:
            ymax = ymindest + h_dst
            if ymax < h_dst:
                ymax = np.abs(ymindest) + conners[2][1] 
        else:
            ymax = conners[2][1]

    # Determine the xmin
    if conners[0][0] < conners[3][0]:
        xmin = conners[3][0] - conners[0][0]
    else:
        xmin = conners[0][0] - conners[3][0]

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)

    pano = panorama[ymin:ymax, xmin:, :]

    return pano

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else 0

def main(args):

    output_file = 'image_names.txt'
    os.makedirs(f"{args.folder}/tmp", exist_ok=True)

    images = sorted([f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))], key=extract_number)
    print(images)


    for i in range(len(images)-1):


        print(f"Processing image {i+1} and its previous image")
        print(f"------------------------------------------")

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

        # check if its neccesary to resize the images
        if i == 0:
            im1 = cv2.imread(f"{args.folder}/tmp/{images[i]}")
        else:
            im1 = cv2.imread(f"{args.folder}/tmp/{path_panorama.split('/')[-1]}")

        im2 = cv2.imread(f"{args.folder}/tmp/{images[i+1]}")

        if im1.shape[0] > 2000 or im1.shape[1] > 2000:
            im1 = cv2.resize(im1, (int(im1.shape[1]*0.5), int(im1.shape[0]*0.5)))
            cv2.imwrite(f"{args.folder}/tmp/{images[i]}", im1)
            cv2.imwrite(f"{args.folder}/{images[i]}", im1)
        
        if im2.shape[0] > 2000 or im2.shape[1] > 2000:
            im2 = cv2.resize(im2, (int(im2.shape[1]*0.5), int(im2.shape[0]*0.5)))
            cv2.imwrite(f"{args.folder}/tmp/{images[i+1]}", im2)
            cv2.imwrite(f"{args.folder}/{images[i+1]}", im2)

        # Run the match_pairs.py script
        run_match_pairs_script(args.folder ,output_file)

        # Remove the temporary images
        if i == 0:
            os.remove(f"{args.folder}/tmp/{images[i]}")
        else:
            os.remove(f"{args.folder}/tmp/{path_panorama.split('/')[-1]}")
        os.remove(f"{args.folder}/tmp/{images[i+1]}")

        # Load the matching keypoints
        if i == 0:
            npz_path = f"{os.path.splitext(images[i+1])[0]}_{os.path.splitext(images[i])[0]}_matches.npz"
        else:
            npz_path = f"{os.path.splitext(images[i+1])[0]}_{os.path.splitext(path_panorama.split('/')[-1])[0]}_matches.npz"
        point_set2, point_set1 = loadNPZ(npz_path, args.folder)

        # Find the homography matrix
        H, _ = cv2.findHomography(point_set1, point_set2)


        # Load images to warp
        im_right = cv2.imread(f"{args.folder}/{images[i+1]}")
            
        if i == 0:
            im_left = cv2.imread(f"{args.folder}/{images[i]}")
        else:
            im_left = cv2.imread(path_panorama)

        # Warp images
        panorama,t,panorama_left,panorama_right , corners = warpImages(im_left, im_right, H)

        # Blend the images
        if args.blend:
            panorama = panoramaBlending(
                panorama_right, panorama_left, t[0], t[1]
            )

        # Crop the panorama
        if args.crop:
            panorama = crop(panorama, im_left.shape[0], corners)


        #verify the size of the uimage and if its greater than 2000x2000 resize mantaining the aspect ratio
        # this is due to limits of match_pairs.py
        if panorama.shape[0] > 2000 or panorama.shape[1] > 2000:
            if panorama.shape[0] > panorama.shape[1]:
                scale_factor = 1800/panorama.shape[0]
            else:
                scale_factor = 1800/panorama.shape[1]
        
            print(f"Scaling factor: {scale_factor}")
            panorama = cv2.resize(panorama, (int(panorama.shape[1]*scale_factor), int(panorama.shape[0]*scale_factor)))

        if i != 0 and i != len(images)-1 and not args.showsteps:
            os.remove(f"{args.folder}/{path_panorama.split('/')[-1]}")

        # Save the panorama
        path_panorama = f"{args.folder}/panorama_{images[i+1].split('.')[0]}_{images[i].split('.')[0]}.jpg"
        cv2.imwrite(path_panorama, panorama)

    os.remove(output_file)
    shutil.rmtree(f"{args.folder}/output")
    shutil.rmtree(f"{args.folder}/tmp")

    print("\n********************************************")
    print("Panorama created successfully")
    print("********************************************")

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch images')
    parser.add_argument('--folder', type=str, help='Folder path containing images', default='./acquisitions')
    parser.add_argument('--crop', default=False, action=argparse.BooleanOptionalAction, help='Crop the panorama to remove black borders')
    parser.add_argument('--blend', default=False, action=argparse.BooleanOptionalAction, help='Blend the images')
    parser.add_argument('--showsteps', default=False, action=argparse.BooleanOptionalAction, help='Show the steps of the panorama process')



    args = parser.parse_args()
    main(args)
