import cv2
import numpy as np


def extract_features_orb(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_features_sift(image):
    # Check if SIFT is available (depends on your OpenCV version)
    if hasattr(cv2, 'SIFT'):
        sift = cv2.SIFT_create()
    else:
        # Older OpenCV versions might require a different method to create a SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features_orb(descriptors1, descriptors2):
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def match_features_sift(descriptors1, descriptors2):
    # Create BFMatcher object with the appropriate norm type
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def associate_depth(keypoints1, keypoints2, matches, depth_image):
    depth_associations = []

    for match in matches:
        # Validate match indices
        if match.queryIdx >= len(keypoints1) or match.trainIdx >= len(keypoints2):
            continue

        # Get the coordinates of the matched keypoints
        x1, y1 = keypoints1[match.queryIdx].pt
        x2, y2 = keypoints2[match.trainIdx].pt

        # Ensure coordinates are within the image boundaries
        if not (0 <= int(x1) < depth_image.shape[1] and 0 <= int(y1) < depth_image.shape[0]):
            continue

        # Get depth for the first keypoint
        depth1 = depth_image[int(y1), int(x1)]

        # If depth is valid, store the match with depth
        if depth1 != 0:  # Assuming 0 as invalid depth
            depth_associations.append((match, depth1))

    return depth_associations


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    # Convert from pixel coordinates to camera-centric coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z])


def calculate_displacements(keypoints1, keypoints2, depth_associations1, depth_associations2, fx, fy, cx, cy):
    displacements = []

    for (match1, depth1), (match2, depth2) in zip(depth_associations1, depth_associations2):
        # Get keypoints from the matches
        kp1 = keypoints1[match1.queryIdx]
        kp2 = keypoints2[match2.trainIdx]

        # Get the coordinates of the keypoints
        u1, v1 = kp1.pt  # feature position in frame 1
        u2, v2 = kp2.pt  # feature position in frame 2

        # Compute 3D positions
        pos1 = pixel_to_3d(u1, v1, depth1, fx, fy, cx, cy)
        pos2 = pixel_to_3d(u2, v2, depth2, fx, fy, cx, cy)

        # Calculate displacement
        displacement = pos2 - pos1
        displacements.append(displacement)

    return displacements






def compute_scaling_factor(curr_rgb, prev_rgb, curr_dp, prev_dp, intrinsics, feature_type = "orb"):
    fx = intrinsics[0]
    fy = intrinsics[1]
    cx = intrinsics[2]
    cy = intrinsics[3]

    # Feature extraction & matching
    if feature_type == "sift":
        keypoints1, descriptors1 = extract_features_sift(prev_rgb)
        keypoints2, descriptors2 = extract_features_sift(curr_rgb)
        # Feature matching
        matches = match_features_sift(descriptors1, descriptors2)
    elif feature_type == 'orb':
        keypoints1, descriptors1 = extract_features_orb(prev_rgb)
        keypoints2, descriptors2 = extract_features_orb(curr_rgb)
        # Feature matching
        matches = match_features_orb(descriptors1, descriptors2)

    # Feature matching
    #matches = match_features(descriptors1, descriptors2)

    # Depth associations
    depth_associations_prev = associate_depth(keypoints1, keypoints2, matches, prev_dp)
    depth_associations_curr = associate_depth(keypoints2, keypoints1, matches, curr_dp)

    # Displacement calculation
    displacements = calculate_displacements(keypoints1, keypoints2, depth_associations_prev, depth_associations_curr, fx, fy, cx, cy)

    avg_dsp = np.mean(displacements, axis=0)

    return avg_dsp






