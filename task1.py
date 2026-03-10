#MSDS24020
#Izba Atif
#Assignment 5

import numpy as np
import utils
import PIL


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - P: Numpy array of shape (3,4) giving the camera projection matrix P

    """
    M = None
    ###########################################################################
    ## Your code here                                                    

    #Set N as the rows of pts2d
    N = pts2d.shape[0]

    #matrix to store
    A = []
    #loop through the size of N
    for i in range(N):
        #set the X,Y,Z from pts3d for each
        X, Y, Z = pts3d[i]
        #set u and v from pts2d
        u, v = pts2d[i]

        #Homogeneous 3D point. Add 1 as it needs it
        Xh = [X, Y, Z, 1]

        #Projection equation: [u,v,1]T=M[X,Y,Z,1]T
        
        #add  to the matrix A
        #[ 0  0  0  0   -Xh   v*Xh ]
        A.append([0, 0, 0, 0, -Xh[0], -Xh[1], -Xh[2], -Xh[3],
                  v*Xh[0], v*Xh[1], v*Xh[2], v*Xh[3]])

        #[-Xh   0    u*Xh ]
        A.append([-Xh[0], -Xh[1], -Xh[2], -Xh[3], 0, 0, 0, 0,
                  u*Xh[0], u*Xh[1], u*Xh[2], u*Xh[3]])

    #convert A to a numpy array
    A = np.array(A)

    #Solve A m = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    #eigenvector with smallest singular value
    #has 12 values (because projection matrix has 12 unknowns)
    m = Vt[-1]        
    #Convert to 3×4 matrix
    M = m.reshape(3, 4)

    #Normalize
    #We can scale matrix M by any non-zero value (homogeneous).
    #So we scale it so that the bottom-right entry becomes 1.
    if abs(M[2, 3]) > 1e-8:
        M = M / M[2, 3]

    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M


if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    res= find_projection(pts2d, pts3d)
    print(res)

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """

