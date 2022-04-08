# @Time : 2020/3/7 16:09
# @Author : Shuai Han
# @Email : DrJasonhan@163.com

"""
This is an example shows how to use Fisher distribution to fit orientation of rock fractures.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mplstereonet


def scatter_3d(fig, x, y, z):
    """
    scatter with a specific format

    :param fig: axes3d
    :param x: array
    :param y: array
    :param z: array
    :return: None
    """
    fig.scatter(x, y, z)
    fig.set_xlabel("x")
    fig.set_ylabel("y")
    fig.set_zlabel("z")
    fig.set_xlim3d(-1, 1)
    fig.set_ylim3d(-1, 1)
    fig.set_zlim3d(-1, 1)
    fig.set_aspect("equal")


def rotate_points(lon, lat, resultant_orien):
    """
    rotate the resultant vector to the north dirction (y-axis), and adjust all the pole according to the resultant vector

    :param lon: array
        original longitude
    :param lat:array
        original latitude
    :param result_v_lon: float
        lon of result vector
    :param result_v_lat: float
        lat of resultant vector
    :return:
    lon_adj: array
        adjusted lon
    lat_adj:
        adjusted lat
    """
    result_v_lon, result_v_lat = mplstereonet.pole(resultant_orien[0], resultant_orien[1])
    lon_adj, lat_adj = mplstereonet.stereonet_math._rotate(
        lon * 180 / np.pi, lat * 180 / np.pi, -result_v_lon * 180 / np.pi, axis='z')
    lon_adj, lat_adj = mplstereonet.stereonet_math._rotate(
        lon_adj * 180 / np.pi, lat_adj * 180 / np.pi, 90 - result_v_lat * 180 / np.pi, axis='y')
    return lon_adj, lat_adj


def rotate_points_back(lon, lat, resultant_orien):
    """
    rotate the simulated vectors back to restore them

    :param lon:
    :param lat:
    :param mean_vector_lon:
    :param mean_vector_lat:
    :return:
    """
    result_v_lon, result_v_lat = mplstereonet.pole(resultant_orien[0], resultant_orien[1])
    lon_adj, lat_adj = mplstereonet.stereonet_math._rotate(
        lon * 180 / np.pi, lat * 180 / np.pi, result_v_lon * 180 / np.pi, axis='z')
    lon_adj, lat_adj = mplstereonet.stereonet_math._rotate(
        lon_adj * 180 / np.pi, lat_adj * 180 / np.pi, result_v_lat * 180 / np.pi - 90, axis='y')
    return lon_adj, lat_adj


def fisher_pdf(orientation, kappa, resultant_orien):
    """
    calculate the pdf of orientation

    :param orientation:
        -arraylike (strike, dip) in degree
    :param kappa:
        -fisher constant
    :param resultant_orien:
        resultant orientation
    :return: float arraylike
    """
    lon, lat = mplstereonet.pole(orientation[0], orientation[1])
    lon_adj, lat_adj = rotate_points(lon, lat, resultant_orien)
    lat_adj = lat_adj * np.pi / 180
    f = kappa * np.sin(lat_adj) * np.exp(kappa * np.cos(lat_adj)) / (2 * np.pi * (np.exp(kappa) - 1))
    return f


def fisher_cdf(theta, kappa):
    """
    calculate the cdf of fisher distribution

    :param theta: float
        -theta in radians
    :param kappa:
        -fisher constant
    :return:F
        -cdf of fisher distribution
    """
    F = np.exp(kappa * np.cos(theta)) / (np.exp(kappa) - np.exp(-kappa))
    return F


def inverse_fisher_cdf(y, kappa):
    """
    inverse of cdf of fisher distribution

    :param y: array
        uniform distribution
    :param kappa: int
        fisher constant
    :return: theta
    """
    theta = np.arccos(np.log((np.exp(kappa) - np.exp(-kappa)) * y) / kappa)
    return theta


def fisher_rvs(kappa, data_size, resultant_orien, form='azimuth'):
    """
    randomly sampling vectors

    :param kappa: float
        -fisher constant
    :param data_size: int
        -the length of simulated vectors
    :param resultant_orien: (float, float)
        -the strike and dip of resultant orientation
    :param form:'azimuth' or 'geographic'
        -'azimuth' generate samples in the format of (strike, dip)
        -'geographic' generate samples in the format of (lon, lat)
    :return:
    """
    y = np.random.uniform(0, 1, data_size)
    sampling_lat = (np.pi / 2 - inverse_fisher_cdf(y, kappa))
    sampling_lon = np.random.uniform(-np.pi, np.pi, 100)

    rotate_back_lon, rotate_back_lat = rotate_points_back(sampling_lon, sampling_lat, resultant_orien)
    sampling_strike, sampling_dip = mplstereonet.geographic2pole(rotate_back_lon, rotate_back_lat)
    if form is 'azimuth':
        return sampling_strike, sampling_dip
    elif form is 'geographic':
        return rotate_back_lon, rotate_back_lat
    else:
        raise ValueError("Error format!")


def fisher_fit(strike, dip):
    """
    fit (strike, dip) dataset using fisher distribution

    :param strike: array
    :param dip: array
    :return:
    kappa: float
        -fisher constant
    resultant_orien: (float, float)
        -resultant orientation
    """
    plunge, bearing = mplstereonet.pole2plunge_bearing(strike, dip)
    resultant_v, (r_value, theta_confi, kappa) = mplstereonet.find_fisher_stats(plunge, bearing)
    resultant_orien = mplstereonet.plunge_bearing2pole(resultant_v[0], resultant_v[1])
    return kappa, resultant_orien

if __name__ == '__main__':
    # read data
    data = pd.read_excel('../Data/Orientation(rockmech)_set2.xlsx')
    strike = data['strike'].values
    dip = data['dip'].values
    # calculate the resultant vector, resultant orientation, and fisher parameters
    kappa, resultant_orien = fisher_fit(strike, dip)

    # project all the points and the resultant vector on a stereonet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    ax.pole(strike, dip, 'g^', markersize=6)
    ax.pole(resultant_orien[0], resultant_orien[1], 'b^', markersize=6)
    ax.grid()
    plt.show()

    # # %%
    # # calculate the lon and lat of original vectors
    # lon, lat = mplstereonet.pole(strike, dip)
    # lon_adj, lat_adj = rotate_points(lon, lat, resultant_orien)
    #
    #
    # fig = plt.figure()
    # fig = Axes3D(fig)
    # # project original vector on a sphere
    # p_x, p_y, p_z = mplstereonet.stereonet2xyz(lon, lat)
    # scatter_3d(fig, p_x, p_y, p_z)
    # # project rotated vectors on the sphere
    # p_x, p_y, p_z = mplstereonet.stereonet2xyz(lon_adj, lat_adj)
    # scatter_3d(fig, p_x, p_y, p_z)


    # simulate 100 samples, and project original vectors and simulated vectors on a stereonet
    sampling_strike, sampling_dip = fisher_rvs(kappa, 100, resultant_orien)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    ax.pole(strike, dip, 'g^', markersize=6)
    ax.pole(sampling_strike, sampling_dip, 'r^', markersize=6)
    ax.grid()
    plt.show()

    # simulate 100 samples, and project both simulated vectors and original vectors on a sphere
    sampling_lon, sampling_lat = fisher_rvs(kappa, 100, resultant_orien, form='geographic')
    # calculate the lon and lat of original vectors
    lon, lat = mplstereonet.pole(strike, dip)
    fig = plt.figure()
    fig = Axes3D(fig)
    p_x, p_y, p_z = mplstereonet.stereonet2xyz(sampling_lon, sampling_lat)
    scatter_3d(fig, p_x, p_y, p_z)
    p_x, p_y, p_z = mplstereonet.stereonet2xyz(lon, lat)
    scatter_3d(fig, p_x, p_y, p_z)

    # calculate the pdf of (strike, dip)
    pdf = fisher_pdf((strike, dip), kappa, resultant_orien)
