#MSDS24020
#Izba Atif
#Assignment 5

from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import os
import argparse

# Normalize points
def normalize_points(pts):
    '''
    Computes a normalization transformation to improve numerical stability for fundamental matrix estimation.
    mean: average x and y coordinates.
    std: average standard deviation of x and y.
    T: similarity transform matrix for normalization:
        Scales points by 1/std and centers them at the origin.
        pts_h = homogenize(pts): convert points to homogeneous [x, y, 1].
        pts_norm = (T @ pts_h.T).T: apply normalization.
    Returns normalized points and the transformation matrix T
    '''
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0).mean()
    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    pts_h = homogenize(pts)
    pts_norm = (T @ pts_h.T).T
    return pts_norm, T

def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    F = None
    ###########################################################################
    # Your code here
    #number of point correspondences.                                                    #
    N = pts1.shape[0]

    #normalize
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    #Builds the linear system A f = 0 using the 8-point algorithm.
    #Each row corresponds to one correspondence (pts1[i], pts2[i]
    A = np.zeros((N, 9))
    for i in range(N):
        u1, v1 = pts1_norm[i, :2]
        u2, v2 = pts2_norm[i, :2]
        A[i] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]

    #Solve A f = 0 via SVD.
    #Last row of Vt is solution f, reshaped into 3x3 fundamental matrix.
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    #fundamental matrix must have rank 2.
    #Set smallest singular value to 0 to enforce this.
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0
    F_norm = U @ np.diag(S) @ Vt

    #Denormalize F to original coordinates using the normalization transforms.
    F = T2.T @ F_norm @ T1

    #                             END OF YOUR CODE                            #
    ###########################################################################
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # e2 is nullspace of F (F e2 = 0).
    _, _, Vt = np.linalg.svd(F)
    e2 = Vt[-1]
    if np.abs(e2[2]) > 1e-8:
        e2 /= e2[2]   # normalize if not zero
    # else leave it as is (point at infinity)

    #Epipole in image 1 is nullspace of F.T (F.T e1 = 0).
    _, _, Vt = np.linalg.svd(F.T)
    e1 = Vt[-1]
    if np.abs(e1[2]) > 1e-8:
        e1 /= e1[2]

    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # Compute essential matrix
    E = K2.T @ F @ K1

    #Decompose E into possible rotations R1, R2 and translation t
    R1, R2, t = cv2.decomposeEssentialMat(E)

    #First camera projection matrix: [I | 0] in camera coordinates.
    M1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))

    #Four possible M2 matrices
    M2_candidates = [
        K2 @ np.hstack((R1,  t)),
        K2 @ np.hstack((R1, -t)),
        K2 @ np.hstack((R2,  t)),
        K2 @ np.hstack((R2, -t)),
    ]

    #Convert points to homogeneous coordinates
    pts1_h = homogenize(pts1).T[:2]
    pts2_h = homogenize(pts2).T[:2]

    best_pcd = None
    best_count = -1

    #Triangulate 3D points for each candidate M2.
    #Convert from homogeneous to Cartesian coordinates.
    for M2 in M2_candidates:
        X_h = cv2.triangulatePoints(M1, M2, pts1_h, pts2_h).T
        X = dehomogenize(X_h)

        # Count points with positive depth in both cameras
        X1 = (M1 @ X_h.T).T
        X2 = (M2 @ X_h.T).T

        count = np.sum((X1[:,2] > 0) & (X2[:,2] > 0))

        #Keep 3D points from the candidate with most points in front of cameras.
        if count > best_count:
            best_count = count
            best_pcd = X



    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_pcd


if __name__ == '__main__':
    #parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True,
                        help="Path to task23 folder containing datasets")
    parser.add_argument("--output_folder", required=True,
                        help="Path to save results")

    args = parser.parse_args()

    #create dir
    os.makedirs(args.output_folder, exist_ok=True)

    names = os.listdir(args.input_folder)

    #loop through inputs and load the data
    for name in names:
        folder = os.path.join(args.input_folder, name)
        print("Processing:", name)

        # Load data
        img1 = cv2.imread(os.path.join(folder, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img2 = cv2.imread(os.path.join(folder, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        data = np.load(os.path.join(folder, "data.npz"))
        pts1 = data["pts1"].astype(float)
        pts2 = data["pts2"].astype(float)
        K1 = data["K1"]
        K2 = data["K2"]

        #Fundamental matrix
        F = find_fundamental_matrix(img1.shape, pts1, pts2)
        np.savetxt(os.path.join(args.output_folder, f"F_{name}.txt"), F)

        #Epipoles
        e1, e2 = compute_epipoles(F)
        np.savetxt(os.path.join(args.output_folder, f"epipoles_{name}.txt"),
                   np.vstack([e1, e2]))

        #Epipolar lines
        draw_epipolar(
            img1, img2, F, pts1, pts2,
            epi1=e1, epi2=e2,
            filename=os.path.join(args.output_folder, f"epi_{name}.png")
        )

        #Triangulation
        pcd = find_triangulation(K1, K2, F, pts1, pts2)
        np.savetxt(os.path.join(args.output_folder, f"pcd_{name}.txt"), pcd)

        #Visualize
        visualize_pcd( pcd, filename=os.path.join(args.output_folder, f"pcd_{name}.png"))


        # you can check against this
        # FCheck, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

        #######################################################################                                              #
        #######################################################################
