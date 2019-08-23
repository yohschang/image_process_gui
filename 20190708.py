# import os
# import cv2
# import numpy as np
# import pandas as pd
# from btimage import check_file_exist
from btimageorigin import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo, Fov, WorkFlow, AnalysisCellFeature
# from btimage import TimeLapseCombo
# import glob
from matplotlib import pyplot as plt
# from os import getenv
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from databaseORM import RetinalPigmentEpithelium
# from scipy.ndimage.filters import gaussian_filter1d

#
# def normalize(array):
#     max_value = max(array)
#     min_value = min(array)
#     return list(map(lambda old: (old - min_value) / (max_value - min_value), array))
#
#
# def moving_average(array, moving_window):
#     array = np.convolve(array, np.ones((moving_window,)) / moving_window, mode='same')
#     return array
#
#
# def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
#     """
#     Calculates the exponential moving average over a vector.
#     Will fail for large inputs.
#     :param data: Input data
#     :param alpha: scalar float in range (0,1)
#         The alpha parameter for the moving average.
#     :param offset: optional
#         The offset for the moving average, scalar. Defaults to data[0].
#     :param dtype: optional
#         Data type used for calculations. Defaults to float64 unless
#         data.dtype is float32, then it will use float32.
#     :param order: {'C', 'F', 'A'}, optional
#         Order to use when flattening the data. Defaults to 'C'.
#     :param out: ndarray, or None, optional
#         A location into which the result is stored. If provided, it must have
#         the same shape as the input. If not provided or `None`,
#         a freshly-allocated array is returned.
#     """
#     data = np.array(data, copy=False)
#
#     if dtype is None:
#         if data.dtype == np.float32:
#             dtype = np.float32
#         else:
#             dtype = np.float64
#     else:
#         dtype = np.dtype(dtype)
#
#     if data.ndim > 1:
#         # flatten input
#         data = data.reshape(-1, order)
#
#     if out is None:
#         out = np.empty_like(data, dtype=dtype)
#     else:
#         assert out.shape == data.shape
#         assert out.dtype == dtype
#
#     if data.size < 1:
#         # empty input, return empty array
#         return out
#
#     if offset is None:
#         offset = data[0]
#
#     alpha = np.array(alpha, copy=False).astype(dtype, copy=False)
#
#     # scaling_factors -> 0 as len(data) gets large
#     # this leads to divide-by-zeros below
#     scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
#                                dtype=dtype)
#     # create cumulative sum array
#     np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
#                 dtype=dtype, out=out)
#     np.cumsum(out, dtype=dtype, out=out)
#
#     # cumsums / scaling
#     out /= scaling_factors[-2::-1]
#
#     if offset != 0:
#         offset = np.array(offset, copy=False).astype(dtype, copy=False)
#         # add offsets
#         out += offset * scaling_factors[1:]
#
#     return out
#
#
# def first_derivatives(array, dx=1):
#     return np.diff(array)/dx




root_path = "D:\\1LAB\\timelapse\\interfereogram\\"
####################################################################################

# TimeLapseCombo(root_path=root_path).combo(target=25, save=False, strategy="try", sp=(1, -2), bg=(0, 3))

####################################################################################
# Fov(root_path).run()

###################################################################################
# label and match
#
current_target = 2
after = CellLabelOneImage(root_path, target=current_target).run(adjust=True, plot_mode=False, load="first", save_water=False)

plt.figure()
plt.imshow(after, cmap='jet')
plt.show()
output = PrevNowCombo(root_path).combo(now_target=current_target, save=False)
# ####################################################################################
# analysis

# AnalysisCellFeature(root_path).image_by_image(db_save=False, png_save=False, plot_mode=False)


####################################################################################
# # peek the data
# passward = getenv("DBPASS")
# engine = create_engine('mysql+pymysql://BT:' + passward + '@127.0.0.1:3306/Cell')
# Session = sessionmaker(bind=engine, autoflush=False)
# sess = Session()
#
# a = sess.query(RetinalPigmentEpithelium).all()
# print(len(a))
#
# sql = '''select * from retinalpigmentepithelium;'''
# df = pd.read_sql_query(sql, engine)

# target = 40
# a = sess.query(RetinalPigmentEpithelium).filter(RetinalPigmentEpithelium.year == 2019
#                                                 , RetinalPigmentEpithelium.month == 7
#                                                 , RetinalPigmentEpithelium.day == 8
#                                                 , RetinalPigmentEpithelium.label == target).order_by(RetinalPigmentEpithelium.id.asc()).all()
# mean_optical_height = []
# phase_mean = []
# phase_std = []
# cir = []
# distance_coef = []
# area = []
#
# for cell in a:
#     print(cell.id)
#     phase_mean.append(cell.phase_mean)
#     mean_optical_height.append(cell.mean_optical_height)
#     phase_std.append(cell.phase_std)
#     cir.append(cell.circularity)
#     distance_coef.append(cell.distance_coef)
#     area.append(cell.area)
#     print(cell.img_path)
#
#
# mean_optical_height = normalize(mean_optical_height)
# phase_std = normalize(phase_std)
# cir = normalize(cir)
# distance_coef = normalize(distance_coef)
# area = normalize(area)
#
# # plt.figure()
# # plt.title("smoothing method")
# # plt.plot(mean_optical_height, label="original")
# # plt.plot(gaussian_filter1d(mean_optical_height, 1), label="gaussian")
# # plt.plot(moving_average(mean_optical_height, 3), label="simple moving average")
# # plt.plot(ewma_vectorized(mean_optical_height, 0.5), label="exponential moving average")
# # plt.legend()
# # plt.show()
# #
# # plt.figure()
# # plt.title("1' derivative with different smoothing methods")
# # plt.plot(first_derivatives(gaussian_filter1d(mean_optical_height, 1)), label="gaussian")
# # plt.plot(first_derivatives(moving_average(mean_optical_height, 3)), label="simple moving average")
# # plt.plot(first_derivatives(ewma_vectorized(mean_optical_height, 0.5)), label="exponential moving average")
# # plt.legend()
# # plt.show()
#
# # plt.figure()
# # plt.plot(moving_average(mean_optical_height,3))
# # for i in np.arange(0, 1, 0.1):
# #
# #     plt.plot(ewma_vectorized(normalize(mean_optical_height), i))
# #
# # plt.show()
#
#
# plt.figure()
# plt.title(str(target))
# plt.plot(phase_mean, label="phase mean")
# plt.plot(phase_std, label="phase std")
# plt.plot(cir, label="cir")
# # plt.plot(distance_coef, label="distance_coef")
# plt.plot(area, label="area")
# plt.xlabel("frame")
# plt.ylabel("au")
# plt.legend()
# plt.show()
#
#
# ## Cannot plot all the cell because of their different apoptosis time.
# # plt.figure()
# # for label in range(2, 90):
# #     a = sess.query(RetinalPigmentEpithelium).filter(RetinalPigmentEpithelium.year == 2019
# #                                                     , RetinalPigmentEpithelium.month == 7
# #                                                     , RetinalPigmentEpithelium.day == 8
# #                                                     , RetinalPigmentEpithelium.label == label).order_by(RetinalPigmentEpithelium.id.asc()).all()
# #
# #     if len(a) == 30:
# #         phase_mean = []
# #         for cell in a:
# #             phase_mean.append(cell.phase_mean)
# #         phase_mean = normalize(phase_mean)
# #         plt.plot(phase_mean)
# #
# # plt.show()
#
# engine.dispose()
#
