# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi, eta): # Here i have calculated the shape-function as told in the notes. I then return the 4 functions in an array.
    """
    Calculates shape functions evaluated at xsi, eta
    """
    # ----- Shape functions -----
    # TODO: fill inn values of the  shape functions
    N1 = (1 - xsi)*(1 - eta)/4
    N2 = (1 + xsi)*(1 - eta)/4
    N3 = (1 + xsi)*(1 + eta)/4
    N4 = (1 - xsi)*(1 + eta)/4

    N = np.array([N1, N2, N3, N4])
    return N

def quad4_shapefuncs_grad_xsi(xsi, eta): # Here we derivate the shapefunction as given above with regards to xsi. We have done it manually,
    #as the professor said it wasnt necesarry to use a pycalculation.
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi
    N1x = (eta-1)/4
    N2x = (1-eta)/4
    N3x = (1+eta)/4
    N4x = (-eta-1)/4

    Ndxi = np.array([N1x, N2x, N3x, N4x])
    return Ndxi


def quad4_shapefuncs_grad_eta(xsi, eta): # Here we derivate the shapefunction as given above with regards to eta.
    """
    Calculates derivatives of shape functions wrt. eta
    """
    # ----- Derivatives of shape functions with respect to eta -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi
    N1e = (xsi-1)/4
    N2e = (-xsi-1)/4
    N3e = (xsi+1)/4
    N4e = (1-xsi)/4

    Ndeta = np.array([N1e, N2e, N3e, N4e])
    return Ndeta




def quad4e(ex, ey, D, thickness, eq=None):
    """
    Calculates the stiffness matrix for a 8 node isoparametric element in plane stress

    Parameters:

        ex  = [x1 ... x4]           Element coordinates. Row matrix
        ey  = [y1 ... y4]
        D   =           Constitutive matrix
        thickness:      Element thickness
        eq = [bx; by]       bx:     body force in x direction
                            by:     body force in y direction

    Returns:

        Ke : element stiffness matrix (8 x 8)
        fe : equivalent nodal forces (4 x 1)

    """
    t = thickness

    if eq is 0:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            #TODO: Calculate Jacobian, inverse Jacobian and determinant of the Jacobian
            J = G @ H # here we determine the jacobi, as well to the inverse.
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0] #sets up the derivated with regards to x
            dNdy = dN[1] #sets up the derivate with regards to y, getting it from the array.

            # Strain displacement matrix calculated at position xsi, eta
            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            B = np.array([[dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0],
                          [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3]],
                          [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3]]])

            # the displacement-interpolation on the xsi and eta positions.
            #TODO: Fill out correct values for displacement interpolation xsi and eta
            N2 = np.array([[N1[0], 0, N1[1], 0, N1[2], 0, N1[3], 0],
                           [0, N1[0], 0, N1[1], 0, N1[2], 0, N1[3]]])

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    return Ke, fe  # Returns stiffness matrix and nodal force vector

def quad9_shapefuncs(xsi, eta):

    N1 = (xsi-xsi**2)*(eta-eta**2)/4
    N2 = (xsi+xsi**2)*(-eta+eta**2)/4
    N3 = (xsi+xsi**2)*(eta+eta**2)/4
    N4 = (-xsi+xsi**2)*(eta+eta**2)/4
    N5 = -(1-xsi**2)*(eta-eta**2)/2
    N6 = (xsi+xsi**2)*(1-eta**2)/2
    N7 = (1-xsi**2)*(eta+eta**2)/2
    N8 = -(xsi-xsi**2)*(1-eta**2)/2
    N9 = (1-eta**2)*(1-xsi**2)

    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9])

    return N

def quad9_shapefuncs_grad_xsi(xsi, eta):

    N1x = (2*xsi - 1)*(eta - 1)*eta/4
    N2x = (2*xsi + 1)*(eta - 1)*eta/4
    N3x = (2*xsi + 1)*(eta + 1)*eta/4
    N4x = (2*xsi - 1)*eta*(eta + 1)/4
    N5x = -xsi*(eta - 1)*eta
    N6x = (2*xsi + 1)*(eta*eta - 1)/(-2)
    N7x = -xsi*eta*(eta + 1)
    N8x = (2*xsi - 1)*(eta*eta - 1)/(-2)
    N9x = 2*xsi*(eta**2 - 1)

    Ndxsi = np.array([N1x, N2x, N3x, N4x, N5x, N6x, N7x, N8x, N9x])

    return Ndxsi

def quad9_shapefuncs_grad_eta(xsi, eta):

    N1e = (xsi - 1)*xsi*(2*eta - 1)/4
    N2e = xsi*(xsi + 1)*(2*eta - 1)/4
    N3e = xsi*(xsi + 1)*(2*eta + 1)/4
    N4e = (xsi - 1)*xsi*(2*eta + 1)/4
    N5e = (xsi**2 - 1)*(2*eta -1)/(-2)
    N6e = (-1)*(xsi + 1)*eta
    N7e = (xsi**2 - 1)*(2*eta + 1)/(-2)
    N8e = -(xsi - 1)*xsi*eta
    N9e = 2*(xsi**2 - 1)*eta

    Ndeta = np.array([N1e,N2e,N3e,N4e,N5e,N6e,N7e,N8e,N9e])

    return Ndeta


def quad9e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a four node membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    Ke = np.array(np.zeros((18,18)))
    fe = np.array(np.zeros((18,1)))

    # TODO: fill out missing parts (or reformulate completely) have not done

    if eq is 0:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)
            N1    = quad9_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            #TODO: Calculate Jacobian, inverse Jacobian and determinant of the Jacobian
            J = G @ H # here we determine the jacobi, as well to the inverse.
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0] #sets up the derivated with regards to x
            dNdy = dN[1] #sets up the derivate with regards to y, getting it from the array.

            # Strain displacement matrix calculated at position xsi, eta
            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            B = np.array([[dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0, dNdx[4], 0, dNdx[5], 0, dNdx[6], 0, dNdx[7], 0, dNdx[8], 0],
                          [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3], 0, dNdy[4], 0, dNdy[5], 0, dNdy[6], 0, dNdy[7], 0, dNdy[8]],
                          [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3], dNdy[4], dNdx[4], dNdy[5], dNdx[5], dNdy[6], dNdx[6], dNdy[7], dNdx[7], dNdy[8], dNdx[8]]])

            # the displacement-interpolation on the xsi and eta positions.
            #TODO: Fill out correct values for displacement interpolation xsi and eta
            N2 = np.array([[N1[0], 0, N1[1], 0, N1[2], 0, N1[3], 0, N1[4], 0, N1[5], 0, N1[6], 0, N1[7], 0, N1[8], 0],
                           [0, N1[0], 0, N1[1], 0, N1[2], 0, N1[3], 0, N1[4], 0, N1[5], 0, N1[6], 0, N1[7], 0, N1[8]]])

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * th * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * th * gw[iGauss] * gw[jGauss]

    if eq is None:
        return Ke
    else:
        return Ke, fe




  