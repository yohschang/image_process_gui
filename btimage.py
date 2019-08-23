"""
Purpose: BT thesis
author: BT
Date: 20190703

class:
    WorkFlow: Create the structure of folder for analyzing RPE cell
    BT_image: Useful image process
    PhaseRetrieval
    TimeLapseCombo: From interferogram to phase image
    MatchFlourPhase: Alignment for two image
    CellLabelOneImage: Semi-automatic Instance Segmentation
    App: Manually Adjust the label on image
    Sketcher
    PrevNowMatching
    PrevNowCombo: Simple RPE object tracking
    AnalysisCellFeature: Extract the features of RPE
    Fov: Define the Field of View

function:
    check_file_exist
    check_img_size

"""

import cv2
from os import makedirs, listdir, getenv
from os.path import isdir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.exposure import adjust_sigmoid
from random import randint
import glob
import tqdm
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
import sys

# from databaseORM import RetinalPigmentEpithelium
from colorbarforAPP import *
from ConfigRPE import *


def check_file_exist(this_path, text):
    my_file = Path(this_path)
    if not my_file.exists():
        raise OSError("Cannot find " + str(text) + "!")


def check_img_size(image):
    try:
        if image.shape[0] != IMAGESIZE or image.shape[1] != IMAGESIZE:
            raise AssertionError("Image size is not " + str(IMAGESIZE) + " !")
    except:
        raise TypeError("This file is not ndarray!")


class BT_image(object):
    def __init__(self, path):
        check_file_exist(path, "image")
        self.path = path
        self.img = None
        self.name = path.split("\\")[-1]
        self.board = None
        self.flat_board = None
        self.centroid_x = 0
        self.centroid_y = 0
        self.threshold = None
        self.f_domain = None
        self.crop_f_domain = None
        self.raw_f_domain = None
        self.crop_raw_f_domain = None
        self.iff = None
        self.test = None

    def open_image(self, color="g"):
        img = cv2.imread(self.path)
        if img is None:
            raise FileNotFoundError("Cannot open this image!")

        if color == "g":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img

    def opennpy(self):
        self.img = np.load(self.path)

    def open_raw_image(self):
        fd = open(self.path, 'rb')
        rows = IMAGESIZE
        cols = IMAGESIZE
        f = np.fromfile(fd, dtype=np.float32, count=rows * cols)
        im_real = f.reshape((rows, cols))
        fd.close()
        self.img = im_real

    def phase2int8(self):
        self.img[self.img >= MAXPHASE] = MAXPHASE
        self.img[self.img <= MINPHASE] = MINPHASE
        max_value = self.img.max()
        min_value = self.img.min()
        image_rescale = (self.img - min_value) * 255 / (max_value - min_value)
        t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
        t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
        self.img = np.uint8(np.round(image_rescale))

    def scaling_image(self, x, y):
        self.img = cv2.resize(self.img, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)

    def beadcenter2croprange(self, x, y):
        return x - 84, y - 84

    def crop(self, x_center, y_center):
        w = 168
        h = 168
        x, y = self.beadcenter2croprange(x_center, y_center)
        self.img = self.img[y:y+h, x:x+w]

    def crop_img2circle_after_crop_it_to_tiny_square(self, centerx, centery):
        """choose the area of bead and append to a list"""
        radius = 48  # pixel
        self.board = np.zeros((self.img.shape[0], self.img.shape[0]))
        self.flat_board = []

        # for i in range(self.img.shape[0]):
        #     for j in range(self.img.shape[0]):
        #         if (i - centerx)**2 + (j - centery)**2 <= radius**2:
        #             self.board[i, j] = self.img[i, j]
        #             self.flat_board.append(self.img[i, j])
        self.board = self.img

    def crop_img2circle(self, centerx, centery, radius):
        self.flat_board = []
        # self.board = np.zeros((self.img.shape[0], self.img.shape[1]))
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if (i - centerx)**2 + (j - centery)**2 <= radius**2:
                    self.flat_board.append(self.img[i, j])
                    # self.board[i, j] = self.img[i, j]

    def write_image(self, path, image):
        cv2.imwrite(path + self.name.split(".")[0] + ".png", image)

    def plot_it(self, image):
        plt.figure()
        plt.title(self.name.split(".")[0])
        plt.imshow(image, plt.cm.gray, vmax=4, vmin=-0.5)
        # plt.scatter(self.centroid_y, self.centroid_x)
        plt.colorbar()
        plt.show()

    def normalize_after_crop(self):
        background = round(float(np.mean(self.img[:20, :20])), 2)
        self.img = self.img - background

    def find_centroid(self):
        # determine threshold
        thres = np.mean(self.img) + 0.7
        # threshold image
        ret, self.threshold = cv2.threshold(self.img, thres, 0, cv2.THRESH_TOZERO)
        # centroid
        moments = cv2.moments(self.threshold)
        if moments['m00'] != 0:
            self.centroid_y = int(moments['m10'] / moments['m00'])
            self.centroid_x = int(moments['m01'] / moments['m00'])
        else:
            print("Cannot find centroid!")

    def twodfft(self):
        # step 1
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # step 2
        dft_shift = np.fft.fftshift(dft)
        # step 3
        self.raw_f_domain = dft_shift

        # visualize
        self.f_domain = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    def twodifft(self, image):
        # step 6
        f_ishift = np.fft.ifftshift(image)
        # step 7
        img_back = cv2.idft(f_ishift)  # complex ndarray [:,:,0]--> real
        # step 8
        self.iff = np.arctan2(img_back[:, :, 1], img_back[:, :, 0])
        self.test = img_back
        img_back = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])
        return img_back

    def crop_first_order(self, sx, sy, size):

        x_start = IMAGESIZE//2 - size//2
        y_start = 0
        width = size
        height = size

        # find the approximate area to crop
        tem_crop = 20 * np.log(self.f_domain[0:0 + IMAGESIZE//4, (IMAGESIZE//2 - IMAGESIZE//8):(IMAGESIZE//2 - IMAGESIZE//8) + IMAGESIZE//4])
        max_y, max_x = np.unravel_index(np.argmax(tem_crop), tem_crop.shape)
        x_final = x_start + max_x + sx
        y_final = y_start + max_y + sy
        crop_f_domain_test = 20 * np.log(self.f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2])
        self.crop_f_domain = self.f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2]

        # step 4
        self.crop_raw_f_domain = self.raw_f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2]
        return self.crop_f_domain, tem_crop, x_final, y_final, crop_f_domain_test


class WorkFlow(object):
    """Create the structure of folder for analyzing RPE cell"""
    def __init__(self, root):
        self.root = root
        check_file_exist(self.root, "root directory")
        self.phase_npy_path = root + "phase_npy\\"
        self.pic_path = root + "pic\\"
        self.marker_path = root + "marker\\"
        self.afterwater_path = root + "afterwater\\"
        self.analysis_path = root + "analysis\\"
        self.fluor_path = root + "fluor\\"

        # create dir
        self.__create_dir(self.phase_npy_path)
        self.__create_dir(self.pic_path)
        self.__create_dir(self.marker_path)
        self.__create_dir(self.afterwater_path)
        self.__create_dir(self.analysis_path)
        self.__create_dir(self.fluor_path)

        # prepare
        self.kaggle_img_path = KAGGLE_IMG
        self.kaggle_mask_path = KAGGLE_MASK

    def __create_dir(self, path):
        """private"""
        my_file = Path(path)
        if not my_file.exists():
            makedirs(path)


class PhaseRetrieval(object):
    """private class. Please do not instantiate it !"""
    def __init__(self, pathsp, pathbg):
        self.name = pathsp.split("\\")[-1].replace(".bmp", "")
        self.path = pathsp.replace(self.name, "").replace(".bmp", "")
        self.sp = BT_image(pathsp)
        self.bg = BT_image(pathbg)
        self.wrapped_sp = None
        self.wrapped_bg = None
        self.unwarpped_sp = None
        self.unwarpped_bg = None
        self.final_sp = None
        self.final_bg = None
        self.final = None
        self.image_size = IMAGESIZE

    def phase_retrieval(self, sp=(0, 0), bg=(0, 0), strategy="try"):
        sys.stdout = open('file.txt', "a+")
        # open img
        self.sp.open_image()
        self.bg.open_image()
        check_img_size(self.sp.img)
        check_img_size(self.bg.img)

        # FFT
        self.sp.twodfft()
        self.bg.twodfft()

        # ----------------------------------------------------------------
        x, y = 0, 0
        bgx, bgy = 0, 0
        if strategy == "try":
            x, y = self.try_the_position(self.sp)
            bgx, bgy = self.try_the_position(self.bg)
        elif strategy == "cheat":
            x, y = sp[0], sp[1]
            bgx, bgy = bg[0], bg[1]

        # crop real or virtual image
        self.sp.crop_first_order(x, y, IMAGESIZE//4)
        self.bg.crop_first_order(bgx, bgy, IMAGESIZE//4)
        print("sp position: ", (x, y), "bg position: ", (bgx, bgy))

        # iFFT
        self.sp.twodifft(self.sp.crop_raw_f_domain)
        self.bg.twodifft(self.bg.crop_raw_f_domain)
        self.wrapped_sp = self.sp.iff
        self.wrapped_bg = self.bg.iff
        # self.plot_fdomain()

        # unwapping
        self.unwarpped_sp = unwrap_phase(self.wrapped_sp)
        self.unwarpped_bg = unwrap_phase(self.wrapped_bg)

        # ----------------------------------------------------------------

        # shift
        sp_mean = np.mean(self.unwarpped_sp)
        bg_mean = np.mean(self.unwarpped_bg)
        self.unwarpped_sp += np.pi * self.shift(sp_mean)
        self.unwarpped_bg += np.pi * self.shift(bg_mean)

        # resize
        self.final_sp = self.resize_image(self.unwarpped_sp, self.image_size)
        self.final_bg = self.resize_image(self.unwarpped_bg, self.image_size)

        # subtract
        self.final = self.final_sp - self.final_bg

        # m_factor
        diff = M - np.mean(self.final)
        self.final = self.final + diff
        sys.stdout.close()

    def try_the_position(self, bt_obj):
        min_sd = 10000
        mini = 100
        minj = 100
        for i in np.arange(-2, 3, 1):
            for j in np.arange(-2, 3, 1):
                bt_obj.crop_first_order(i, j, IMAGESIZE//4)
                bt_obj.twodifft(bt_obj.crop_raw_f_domain)
                unwrap_ = unwrap_phase(bt_obj.iff)
                buffer_sd = np.std(unwrap_)
                if buffer_sd < min_sd:
                    mini, minj, min_sd = i, j, buffer_sd
        return mini, minj

    def shift(self, sp_mean):
        interval_list = [x - np.pi/2 for x in np.arange(-6 * np.pi, 7 * np.pi, np.pi)]
        i = 0
        for i, interval in enumerate(interval_list):
            if sp_mean < interval:
                break
        shift_pi = 7 - i
        # print(sp_mean, 'so', shift_pi)
        return shift_pi

    def resize_image(self, image, size):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    def plot_fdomain(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        axes[0].imshow(self.sp.f_domain, cmap='gray')
        axes[0].set_title("sp f_domain ")

        axes[1].imshow(self.bg.f_domain, cmap='gray')
        axes[1].set_title("bg f_domain ")

        fig.subplots_adjust(right=1)
        plt.show()

    def plot_sp_bg(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        im = axes[0].imshow(self.unwarpped_sp, cmap='gray', vmin=-20, vmax=20)
        axes[0].set_title("sp")

        im = axes[1].imshow(self.unwarpped_bg, cmap='gray', vmin=-20, vmax=20)
        axes[1].set_title("bg")

        fig.subplots_adjust(right=1)
        cbar_ax = fig.add_axes([0.47, 0.1, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.ioff()
        plt.show()

    def plot_final(self, center=False, num=0):
        """beware of Memory"""
        plt.figure(dpi=200, figsize=(10, 10))
        if center:
            plt.imshow(self.final[1500:1700, 1500:1700], vmin=-1, vmax=MAXPHASE)
        else:
            plt.imshow(self.final, cmap='jet', vmin=-0.5, vmax=3)
        plt.colorbar()
        plt.title("sp - bg"+str(num))
        plt.show()

    def plot_hist(self):
        plt.figure()
        plt.hist(self.final.flatten(), bins=100)
        plt.xlim(-5, 5)
        plt.show()

    def write_final(self, dir_npy):
        np.save(dir_npy + self.name + "_phase.npy", self.final)


class TimeLapseCombo(WorkFlow):
    sys.stdout = open('file.txt', 'w')
    def __init__(self, root_path):
        super().__init__(root_path)
        self.pathsp_list = []
        self.pathbg_list = []
        self.cur_num = 1
        self.SD_threshold = 1.51

    def __read(self):
        sys.stdout = open('file.txt', 'w')
        """ private """
        file_number = len(glob.glob(self.root + "[0-9]*"))
        print("Found ", file_number, "pair image")
        for i in range(1, file_number+1):

            # read one BG at the root dir
            found_bg = glob.glob(self.root + "*.bmp")
            if len(found_bg) < 1:
                raise FileExistsError("BG lost")
            print("BG:", found_bg[0])
            self.pathbg_list.append(found_bg[0])

            # read many SP
            path_cur = self.root + str(i) + "\\"
            check_file_exist(path_cur, " interferogram #" + str(i))
            # find interferogram
            found_file = glob.glob(path_cur + "*.bmp")
            if len(found_file) != 1:
                raise FileExistsError("SP lost or too many SP")
            print("SP:", found_file[0])
            self.pathsp_list.append(found_file[0])
        sys.stdout.close()
    def combo(self, target=-1, save=False, strategy="try", sp=(0, 0), bg=(0, 0)):
        """ target is the number of image """
        self.__read()
        plt.ion()
        if target == -1:
            # combo
            for i, m in zip(range(len(self.pathsp_list)), np.arange(0.3, 0, -0.3/40)):
                pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[0])
                sys.stdout = open('file.txt', 'a+')
                try:
                    pr.phase_retrieval(sp, bg, strategy=strategy)
                    print(str(i), " SD:", np.std(pr.final))
                    if np.std(pr.final) > self.SD_threshold:
                        pr.phase_retrieval(sp, bg, strategy=strategy)
                    pr.plot_final(center=False, num=i)
                    # pr.plot_hist()
                    if save:
                        np.save(self.phase_npy_path + str(self.cur_num) + "_phase.npy", pr.final)
                        self.cur_num += 1
                except TypeError as e:
                    print(i, "th cannot be retrieved ", e)
                sys.stdout.close()

        elif target > 0:
            # specific target
            pr = PhaseRetrieval(self.pathsp_list[target-1], self.pathbg_list[0])
            pr.phase_retrieval(sp, bg, strategy=strategy)
            sys.stdout = open('file.txt', 'a+')
            if np.std(pr.final) > self.SD_threshold:
                pr.phase_retrieval(sp, bg, strategy=strategy)
            pr.plot_final(center=False, num=target)
            # pr.plot_hist()
            pr.plot_sp_bg()
            # pr.plot_fdomain()

            print("phase std: ", np.std(pr.final))
            if save:
                np.save(self.phase_npy_path + str(target) + "_phase.npy", pr.final)
                print(self.phase_npy_path + str(target) + "_phase.npy")
                # pr.write_final(output_dir)
        else:
            raise IndexError("Invalid target number!")
        sys.stdout.close()

class MatchFlourPhase(object):
    def __init__(self, path_phasemap, path_flour):
        
        # read phase image
        check_file_exist(path_phasemap, "phase image")
        im = BT_image(path_phasemap)
        im.open_image()
        self.im = im
        
        # read flour image
        check_file_exist(path_flour, "flour image")
        im_f = BT_image(path_flour)
        im_f.open_image()
        im_f.img = cv2.flip(im_f.img, -1)
        self.im_f = im_f

        # two image diffeerence
        self.m_obj = 28
        self.m = 46.5
        self.viework_pixel = 5.5
        self.point_gray_pixel = 5.5

    def match(self, shift_x, shift_y):
        ratio = (self.m / self.viework_pixel) / (self.m_obj / self.point_gray_pixel)
        new_size = int(self.im_f.img.shape[0] * ratio)
        self.im_f.img = cv2.resize(self.im_f.img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)

        # Photonfocus crop to 3072 * 3072
        b = self.im_f.img.shape[0]
        start = b // 2 - IMAGESIZE // 2
        end = b // 2 + IMAGESIZE // 2
        self.im_f.img = self.im_f.img[start - shift_y: end - shift_y, start - shift_x: end - shift_x]

        # subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        im0 = axes[0].imshow(self.im.img, cmap='gray')
        axes[0].set_title("Phase image", fontsize=30)

        im1 = axes[1].imshow(self.im_f.img, cmap=green)
        axes[1].set_title("Fluorescent image", fontsize=30)

        fig.subplots_adjust(right=1)
        cbar_ax0 = fig.add_axes([0.47, 0.1, 0.02, 0.8])
        cbar_ax1 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
        fig.colorbar(im0, cax=cbar_ax0)
        cbar_ax0.set_title('rad')
        fig.colorbar(im1, cax=cbar_ax1)
        cbar_ax1.set_title('a.u.')
        plt.show()

    def save(self, pathandname):
        np.save(pathandname, self.im_f.img)


class CellLabelOneImage(WorkFlow):
    """Instance segmentation"""

    def __init__(self, root, target=-1):
        super().__init__(root)

        # open image
        self.img = np.load(self.phase_npy_path + str(target) + "_phase.npy")
        check_img_size(self.img)

        self.target = target
        self.img_origin = None
        self.sure_bg = None
        self.sure_fg = None
        self.pre_marker = None
        self.distance_img = None
        self.after_water = None
        self.plot_mode = False

    def run(self, adjust=False, plot_mode=False, load="no", save_water=False):
        self.plot_mode = plot_mode
        self.__phase2uint8()
        self.__smoothing()
        self.__sharpening(0.15, 30)
        self.__adaptive_threshold()
        self.__morphology_operator()
        self.__prepare_bg()
        self.__distance_trans()
        self.__find_local_max()
        self.__watershed_algorithm()
        if adjust:
            if load == "old":
                sys.stdout = open('file.txt', "a+")
                try:
                    # load saved marker
                    print("load saved marker", str(self.target) + "_marker.npy")
                    marker_file = self.marker_path + str(self.target) + "_marker.npy"
                    check_file_exist(marker_file, str(self.target) + "_marker.npy")
                except OSError:
                    # load previous marker
                    print("load previous marker", str(self.target-1) + "_marker.npy")
                    marker_file = self.marker_path + str(self.target-1) + "_marker.npy"
                    check_file_exist(marker_file, str(self.target-1) + "_marker.npy")
                sys.stdout.close()

                sys.stdout = open('file.txt', "a+")
                try:
                    # load previous afterwater
                    print("load previous afterwater", str(self.target-1) + "_afterwater.npy")
                    afterwater_file = self.afterwater_path + str(self.target-1) + "_afterwater.npy"
                    check_file_exist(afterwater_file, str(self.target-1) + "_afterwater.npy")
                except OSError:
                    raise Exception("Must use previous afterwater!")
                sys.stdout.close()

            elif load == "first":
                sys.stdout = open('file.txt', "a+")
                try:
                    marker_file = self.marker_path + str(self.target) + "_marker.npy"
                    check_file_exist(marker_file, str(self.target) + "_marker.npy")
                except OSError:
                    print("No ", str(self.target) + "_marker.npy")
                    marker_file = None
                afterwater_file = None
                sys.stdout.close()

            elif load == "no":
                marker_file = None
                afterwater_file = None
            else:
                raise Exception("invalid argument of load: " + load)
            self.__watershed_manually(marker_file, afterwater_file)

        if save_water:
            sys.stdout = open('file.txt', "a+")
            np.save(self.afterwater_path + str(self.target) + "_afterwater.npy", self.after_water)
            print("saving  ", self.afterwater_path + str(self.target) + "_afterwater.npy")
            sys.stdout.close()

        if self.after_water is None:
            sys.stdout = open('file.txt', "a+")
            raise EOFError("You must go watershed once!")
            sys.stdout.close()
        return self.after_water



    def __plot_gray(self, image, title_str):
        plt.figure()
        plt.title(title_str)
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.show()

    def __phase2uint8(self):
        # plt.figure(dpi=200, figsize=(10, 10))
        # plt.title(str(self.target) + " original image")
        # plt.imshow(self.img, cmap='jet', vmax=2.5, vmin=MINPHASE)
        # plt.axis("off")
        # plt.show(block = False)
        self.img[self.img >= 4] = 0
        self.img[self.img <= -0.5] = -0.5
        max_value = self.img.max()
        min_value = self.img.min()
        image_rescale = (self.img - min_value) * 255 / (max_value - min_value)
        t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
        t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
        self.img = np.uint8(np.round(image_rescale))
        if self.plot_mode:
            self.__plot_gray(self.img, "original image")
        return self.img

    def __smoothing(self):
        self.img = cv2.GaussianBlur(self.img, (7, 7), sigmaX=1)
        # show

    def __sharpening(self, cutoff_value, gain_value):
        if self.plot_mode:
            plt.figure()
            plt.hist(self.img.flatten(), bins=200)
            plt.show()
        # self.img = enhance_contrast(self.img, disk(5))
        self.img = adjust_sigmoid(self.img, cutoff=cutoff_value, gain=gain_value)
        self.img_origin = self.img.copy()
        # show

        x = np.arange(0, 1, 0.01)
        y = 1/(1 + np.exp((gain_value*(cutoff_value - x))))
        if self.plot_mode:
            plt.figure()
            plt.title("Sigmoid Correction (cutoff: 0.08, gain: 18)")
            plt.plot(x, y)
            plt.show()

            plt.figure()
            plt.title("smoothing and sharpening")
            plt.imshow(self.img, cmap='gray')
            plt.show()

    def __adaptive_threshold(self):
        sys.stdout = open('file.txt',"w")
        array_image = self.img.flatten()
        # plt.figure()
        n, b, patches = plt.hist(array_image, bins=200)
        # plt.title("Histogram of phase image")
        # plt.xlabel("gray value")
        # plt.ylabel("number of pixel")
        # plt.show()

        # Adaptive threshold
        b = b[:-1]
        n[b < 70] = 0
        n[b > 220] = 0
        bin_max = np.argmax(n)
        print("bin_max", bin_max)
        max_value = b[bin_max]
        print(max_value)
        threshold = 0.5 * np.sum(array_image) / len(array_image[array_image > max_value])
        print("Adaptive threshold is:", threshold)
        # thresholding
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 1)
        # ret, self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)
        if self.plot_mode:
            self.__plot_gray(self.img, "binary image")
        sys.stdout.close()

    def __morphology_operator(self):
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=4)
        kernel = np.ones((30, 30), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.plot_mode:
            self.__plot_gray(self.img, "morphology image")

    def __prepare_bg(self):
        self.sure_bg = self.img.copy()
        # kernel = np.ones((5, 5), np.uint8)
        # self.sure_bg = np.uint8(cv2.dilate(self.sure_bg, kernel, iterations=5))

    def __distance_trans(self):
        """
                    force watershed algorithm flow to the value distance = 0 but not sure bg,
                    so distance map += 1
                    distance map where the location is sure bg -= 1
                """
        self.img = cv2.distanceTransform(self.img, 1, 5) + 1
        self.img[self.sure_bg == 0] -= 1
        # plt.figure()
        # plt.hist(self.img.flatten(), bins=100)
        # plt.show()
        # self.img = np.power(self.img/float(np.max(self.img)), 0.6) * 255
        # remove too small region
        # self.img[self.img < 50] = 0

        self.distance_img = self.img.copy()
        if self.plot_mode:
            # self.plot_gray(self.img, "dist image")
            plt.figure()
            plt.title("distance transform")
            plt.imshow(self.img, cmap='jet')
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.hist(self.img.flatten(), bins=100)
            plt.show()

    def __find_local_max(self):
        marker = np.zeros((IMAGESIZE, IMAGESIZE), np.uint8)

        # 220 is the size of RPE
        local_maxi = peak_local_max(self.img, indices=False, footprint=np.ones((220, 220)), threshold_abs=20)
        marker[local_maxi == True] = 255
        kernel = np.ones((5, 5), np.uint8)
        marker = np.uint8(cv2.dilate(marker, kernel, iterations=15))

        ret, markers1 = cv2.connectedComponents(marker)
        markers1[self.sure_bg == 0] = 1
        self.pre_marker = np.int32(markers1)
        if self.plot_mode:
            self.__plot_gray(self.pre_marker, "local max image")

    def __watershed_algorithm(self):
        rgb = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        self.after_water = self.pre_marker.copy()
        cv2.watershed(rgb, self.after_water)
        if self.plot_mode:
            self.__plot_gray(self.after_water, "watershed image")
        ###########################################################
        # if no manually adjust, self.after_water is final output #
        ###########################################################

    def __watershed_manually(self, marker_file=None, afterwater_file=None):
        """Implement App"""
        self.img_origin = cv2.cvtColor(self.img_origin, cv2.COLOR_GRAY2BGR)
        self.distance_img = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        sys.stdout = open('file.txt',"a+")
        print(App.__doc__)
        sys.stdout.close()
        sys.stdout = open('file.txt', "a+")
        if marker_file:
            try:
                self.pre_marker = np.load(marker_file)
            except:
                raise FileExistsError("cannot open marker file")

        if afterwater_file:
            try:
                afterwater = np.load(afterwater_file)
            except:
                raise FileExistsError("cannot open afterwater file")
        else:
            afterwater = np.zeros((IMAGESIZE, IMAGESIZE), dtype=np.uint8)


        r = App(self.distance_img, self.pre_marker, self.img_origin, save_path=self.marker_path, cur_img_num=self.target, afterwater=afterwater)
        r.run()
        sys.stdout.close()
        self.after_water = r.m


class Sketcher(object):
    """Private class. Please do not instantiate it !"""
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.diameter = 20
        self.show()
        self.mouse_track = None
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.dests[0])
        cv2.resizeWindow(self.windowname, 640, 640)

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)

        # the track of mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt

        if event == cv2.EVENT_RBUTTONDOWN:
            if self.diameter == 20:
                print("large diameter ^^ ")
                self.diameter = 50
            else:
                print("small diameter :( ")
                self.diameter = 20

        # draw a line
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            # print(self.colors_func())
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, self.diameter)
            self.dirty = True
            self.prev_pt = pt
            self.mouse_track = pt
            self.show()
        else:
            self.prev_pt = None
            self.mouse_track = pt


class App(object):
    """
            Watershed segmentation
            =========
            Keys
            ----
              SPACE - update segmentation
              a    - find which label is available
              s    - save marker.npy
              q    - check two separated regions
              t    - remove current marker
              c    - catch the label on the image
              l    - use input as label number
              right click  - change the diameter of your mouse
              r     - reset
              ESC   - exit and save afterwater.npy
        """
    def __init__(self, fn, existed_marker, show_img, save_path, cur_img_num, afterwater):
        # input parameter
        self.img = fn
        self.markers = existed_marker
        # sketcher image
        jet_afterwater = cv2.applyColorMap(afterwater.astype(np.uint8) * 3, cv2.COLORMAP_JET)
        self.show_img = cv2.addWeighted(show_img, 0.5, jet_afterwater, 0.5, 0.0, dtype=cv2.CV_8UC3)
        # self.show_img = show_img
        self.save_path = save_path
        self.cur_img_num = cur_img_num

        # create parameter
        self.markers_vis = self.show_img.copy()
        self.cur_marker = 1
        self.colors = jet_color
        self.overlay = None
        self.m = None

        # marker pen diameter
        diameter = 20
        self.auto_update = False

        # canvas
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.__get_colors)

    def __get_colors(self):
        pen_color = self.cur_marker*3
        if pen_color > 255:
            pen_color -= 255
        return list(map(int, self.colors[pen_color])), int(self.cur_marker)

    def __watershed(self):

        # because watershed will change m
        self.m = self.markers.copy()

        # watershed algorithm
        cv2.watershed(self.img, self.m)

        # transfer marker to color but remove negative marker
        marker_map = np.maximum(self.m, 0) * 3
        marker_map[marker_map > 255] -= 255
        self.overlay = self.colors[marker_map]

        vis = cv2.addWeighted(self.img, 0.5, self.overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        cv2.namedWindow('watershed', cv2.WINDOW_NORMAL)
        cv2.imshow('watershed', vis)
        cv2.resizeWindow("watershed", 640, 640)
        # plt.close()
        # plt.figure()
        # plt.imshow(self.markers, cmap='jet')
        # plt.show()

    def run(self):
        # init marker
        while True:
            ch = 0xFF & cv2.waitKey(50)

            # Esc
            if ch == 27:
                break

            if ch == ord("0"):
                self.cur_marker = 0
                sys.stdout = open('file.txt',"w")
                print(App.__doc__)
                print('marker: ', self.cur_marker)
                sys.stdout.close()

            if ch in [ord('l'), ord('L')]:
                f = open('file.txt', "r")
                line = f.readlines()
                f.close()
                try :
                    number = int(line[0])
                    sys.stdout = open('file.txt', "w")
                    print(App.__doc__)
                    if 0 <= number <= 90:
                        self.cur_marker = number
                    else:
                        print("invalid label!")
                    print('marker: ', self.cur_marker)
                    sys.stdout.close()
                except ValueError :
                    sys.stdout = open('file.txt', "w")
                    print(App.__doc__)
                    print("cannot find the number")
                    sys.stdout.close()

            if ch in [ord('t'), ord('T')]:
                sys.stdout = open('file.txt', "w")
                print(App.__doc__)
                if self.cur_marker == 1 or self.cur_marker == 0:
                    print("Cannot delete background label or unknown label!")
                else:
                    self.markers[self.markers == self.cur_marker] = 0
                    print('reset: ', self.cur_marker, " in the image")
                sys.stdout.close()

            # update watershed
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.__watershed()
                self.sketch.dirty = False

            if ch in [ord('a'), ord('A')]:
                sys.stdout = open('file.txt', "w")
                print(App.__doc__)
                for label in range(85):
                    if len(self.markers[self.markers == label]) == 0:
                        print(label, " is available")
                sys.stdout.close()

            if ch in [ord('q'), ord('Q')]:
                sys.stdout = open('file.txt', "w")
                print(App.__doc__)
                for label in range(2, 85):
                    black = np.zeros((IMAGESIZE, IMAGESIZE), dtype=np.uint8)
                    black[self.m == label] = 255
                    ret, b = cv2.connectedComponents(black)
                    if ret > 2:
                        print("label ", label, "is too many region!!")
                print("Q: finish checking!")
                sys.stdout.close()

            # reset
            if ch in [ord('r'), ord('R')]:
                # self.markers[:] = 0
                self.markers_vis[:] = self.show_img
                self.sketch.show()

            # save
            if ch in [ord('s'), ord('S')]:
                sys.stdout = open('file.txt', "w")
                print(App.__doc__)
                self.__watershed()
                np.save(self.save_path + str(self.cur_img_num) + "_marker.npy", self.markers)
                print("save marker to ", self.save_path + str(self.cur_img_num) + "_marker.npy")
                sys.stdout.close()

            # catch the marker
            if ch in [ord('c'), ord('C')]:
                sys.stdout = open('file.txt', "w")
                print(App.__doc__)
                print("track the mouse:", self.sketch.mouse_track)
                self.cur_marker = self.markers[self.sketch.mouse_track[1], self.sketch.mouse_track[0]]
                print("marker:", self.cur_marker)
                sys.stdout.close()
        cv2.destroyAllWindows()


class PrevNowMatching(object):
    """ creare the list of linkage"""
    def __init__(self, prev, now):
        self.prev_label_map = prev
        self.now_label_map = now

        # clean label
        self.prev_label_map[self.prev_label_map == -1] = 1
        self.now_label_map[self.now_label_map == -1] = 1

        self.prev_list = []
        self.now_list = []
        self.output = None

        # lost map
        self.lost_map = np.zeros((IMAGESIZE, IMAGESIZE))

    def run(self):
        self.__check_prev_label()
        self.__check_now_label()
        self.__first_round_matching()
        self.__second_round_matching()
        self.__clean_appear()
        return self.output

    def show(self, image, text):
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='jet', vmax=90, vmin=0)
        for i in range(2, 90):
            if len(image[image == i]) != 0:
                image_tem = np.zeros((IMAGESIZE, IMAGESIZE), dtype=np.uint8)
                image_tem[image == i] = 255
                x, y = self.__centroid(image_tem)
                plt.scatter(x, y, s=5, c='g')
                plt.text(x, y, str(i))
        plt.title(text)
        plt.axis("off")
        plt.show()

    def __check_prev_label(self):
        for label in range(90):
            cur_label_num = len(self.prev_label_map[self.prev_label_map == label])
            if 4000 <= cur_label_num <= 1000000:
                # remove too small area and BG area
                self.prev_list.append(label)
                # print("label:", label, "has:", cur_label_num, "pixel")

    def __check_now_label(self):
        for label in range(90):
            cur_label_num = len(self.now_label_map[self.now_label_map == label])
            if 4000 <= cur_label_num <= 1000000:
                # remove too small area and BG area
                self.now_list.append(label)
                # print("label:", label, "has:", cur_label_num, "pixel")

    def __first_round_matching(self):
        """ centroid method """
        self.output = self.now_label_map.copy()
        iterative_label = self.prev_list.copy()
        # find prev label
        for i in range(len(iterative_label)):
            # choose iterative label in box
            label = iterative_label[i]

            # find corresponding label in now
            black = np.zeros((IMAGESIZE, IMAGESIZE))
            black[self.prev_label_map == label] = 255
            x, y = self.__centroid(black)
            corresponded_label = self.now_label_map[y, x]


            # if i == 33:
            #     plt.figure()
            #     plt.imshow(black, cmap='gray', vmax=255, vmin=0)
            #     black[self.now_label_map == corresponded_label] = 100
            #     plt.scatter(x, y, s=20, c="g")
            #     plt.show()

            # print("prev label:", label, "match --> now label: ", corresponded_label)
            sys.stdout = open("file.txt", "a+")
            if corresponded_label != 1:
                # registering corresponding label into new_now_map
                self.output[self.now_label_map == corresponded_label] = label
                # pop corresponded_label
                try:
                    self.prev_list.remove(label)
                except Exception as e:
                    print(str(e))
                    print("Cannot remove ", label, " from disappear list")

                try:
                    self.now_list.remove(corresponded_label)
                except Exception as e:
                    print(str(e))
                    print("Cannot remove ", corresponded_label, " from appear list")

            elif corresponded_label == 1:
                print("prev label:", label, "match BG label !!!!!")

            else:
                print("prev label:", label, "match ", corresponded_label, "what?!")
            sys.stdout.close()

    def __second_round_matching(self):
        """ overlap method"""
        disappear_list = self.prev_list.copy()
        appear_list = self.now_list.copy()
        sys.stdout = open('file.txt' , "a+")
        if appear_list and disappear_list:
            for disappear in disappear_list:
                for appear in appear_list:
                    black = np.zeros((IMAGESIZE, IMAGESIZE))
                    if len(black[(self.prev_label_map == disappear) & (self.now_label_map == appear)]) != 0:
                        # find overlap
                        print("Round 2 : prev label:", disappear, "match --> now label: ", appear)
                        self.output[self.now_label_map == appear] = disappear

                        # remove disappear and appear
                        self.prev_list.remove(disappear)
                        self.now_list.remove(appear)

        # appear
        print("appear: ", self.now_list)
        if self.now_list:
            for i in self.now_list:
                self.lost_map[self.now_label_map == i] = 200

        # disappear
        print("disappear: ", self.prev_list)
        if self.prev_list:
            for i in self.prev_list:
                self.lost_map[self.prev_label_map == i] = 100

        print("finish second round!")
        sys.stdout.close()

    def __clean_appear(self):
        """clean appear cell"""
        for label_appeared in self.now_list:
            self.output[self.now_label_map == label_appeared] = 1

        self.show(self.output, "new")
        plt.figure()
        plt.imshow(self.lost_map, cmap='jet', vmax=255, vmin=0)
        plt.figtext(0.83, 0.5, "g: disappear\no: appear", transform=plt.gcf().transFigure)
        plt.title("lost_map")
        plt.show(block = False)

    def __centroid(self, binary_image):
        """ find centroid"""
        moments = cv2.moments(binary_image)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_y, centroid_x = 0, 0
            print("Cannot find centroid!")
        return centroid_x, centroid_y


class PrevNowCombo(WorkFlow):
    def __init__(self, root):
        super().__init__(root)
        self.prev = None
        self.now = None

    def __read(self, now_target):
        sys.stdout = open('file.txt',"w")
        # load prev and now label map
        prev_file = str(now_target-1) + "_afterwater.npy"
        now_file = str(now_target) + "_afterwater.npy"
        print("load ", prev_file)
        check_file_exist(self.afterwater_path + prev_file, str(prev_file))
        self.prev = np.load(self.afterwater_path + prev_file)
        print("load ", now_file)
        check_file_exist(self.afterwater_path + now_file, str(now_file))
        self.now = np.load(self.afterwater_path + now_file)
        sys.stdout.close()

    def combo(self, now_target=-1, save=False):
        """ now_target is the number of image"""
        self.__read(now_target=now_target)
        # map
        match = PrevNowMatching(self.prev, self.now)
        output = match.run()
        # plot input
        match.show(match.prev_label_map, str(now_target-1) + " prev_label_map")
        match.show(match.now_label_map, str(now_target) + " now_label_map")
        if save:
            sys.stdout = open('file.txt', "a+")
            if match.prev_list:
                print("Cannot save it ! disappear !!")
            else:
                print("revise ", str(now_target) + "_afterwater.npy")
                np.save(self.afterwater_path + str(now_target) + "_afterwater.npy", output)
            sys.stdout.close()


###########################################################################################

class AnalysisCellFeature(WorkFlow):
    """ Find the features for every cell. Store the features in Database"""
    def __init__(self, root):
        super().__init__(root)

        # find
        image_num = len(listdir(self.phase_npy_path))
        print("Found ", image_num, " phase images~")
        label_num = len(listdir(self.afterwater_path))
        print("Found ", label_num, " label images~")

        # find path
        self.phase_img_list = []
        self.label_img_list = []
        for i in range(np.amin([image_num, label_num])):
            self.phase_img_list.append(self.phase_npy_path + str(i+1) + "_phase.npy")
            self.label_img_list.append(self.afterwater_path + str(i+1) + "_afterwater.npy")

        self.analysis_label = []

        # database
        self.dbsave = False
        self.pngsave = False
        self.plot_mode = False
        self.precision = 3
        self.sess = None
        self.engine = None
        self.current_id = 0
        self.date = (2019, 7, 8)
        self.__connect_to_db()

    def image_by_image(self, db_save=False, png_save=False, plot_mode=False):
        self.dbsave = db_save
        self.pngsave = png_save
        self.plot_mode = plot_mode
        # find id
        self.check_last_id()
        for i in tqdm.trange(len(self.phase_img_list)):
            print("image ", str(i+1))
            # an image
            self.__one_by_one(i)
        if self.dbsave:
            self.__db_commit()

    def __one_by_one(self, i):

        # load phase img and label img
        phase_img = np.load(self.phase_img_list[i])
        label_img = np.load(self.label_img_list[i])

        # specify those label we want to analyze
        if i == 0:
            self.__label_analyzed(label_img)

        # for each label
        for label in self.analysis_label:

            # crop the cell
            phase_copy = phase_img.copy()
            phase_copy[label_img != label] = 0
            a = phase_copy[label_img == label]

            # binarization
            binary = phase_img.copy()
            binary[label_img == label] = 255
            binary[label_img != label] = 0
            binary = binary.astype(np.uint8)
            x, y, w, h = cv2.boundingRect(binary)

            # mean
            phase_mean = round(float(np.mean(a)), self.precision)
            # std
            phase_std = round(float(np.std(a)), self.precision)
            # area, circularity
            binary, contours, hierarchy = cv2.findContours(binary,
                                                           cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 1:
                area_list = [cv2.contourArea(contours[idx]) for idx in range(len(contours))]
                max_contour_idx = np.argmax(area_list)
                object_contours = contours[max_contour_idx]
            else:
                object_contours = contours[0]
            epsilon = 0.005 * cv2.arcLength(object_contours, True)
            approx_contour = cv2.approxPolyDP(object_contours, epsilon, True)

            perimeter = cv2.arcLength(approx_contour, True)
            area = cv2.contourArea(approx_contour)
            circularity = round(4 * np.pi * area / perimeter ** 2, self.precision)

            # mean optical height
            height = round(0.532 * phase_mean / 2 / np.pi / (1.37 - 1.33), self.precision)

            # crop image
            crop_binary = binary[y:y + h, x:x + w]
            crop_phase = phase_copy[y:y + h, x:x + w]
            crop_phase[crop_phase >= MAXPHASE] = MAXPHASE
            crop_phase[crop_phase <= MINPHASE] = MINPHASE
            crop_phase = 255 * (crop_phase - MINPHASE) / (MAXPHASE - MINPHASE)
            crop_phase = crop_phase.astype(np.uint8)

            # distance coef
            dis = cv2.distanceTransform(crop_binary, cv2.DIST_L2, 5)
            res = cv2.matchTemplate(crop_phase, dis.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
            distance_coef = round(float(res.max()), self.precision)

            # apoptosis
            apoptosis = False

            print("mean: ", phase_mean, "std", phase_std, "area: ", area, "circularity: ", circularity, "height: ", height, "dis_coef: ", distance_coef)
            features = [phase_mean, phase_std, circularity, area, apoptosis, height, distance_coef]

            if self.plot_mode:
                rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                rgb = cv2.drawContours(rgb, approx_contour, -1, (255, 0, 0), 5)

                plt.figure()
                plt.imshow(rgb)
                plt.show()

                plt.figure()
                plt.imshow(crop_binary, cmap="gray")
                plt.show()

                plt.figure()
                plt.title("uint8")
                plt.imshow(crop_phase, cmap="gray", vmax=255, vmin=0)
                plt.show()

                plt.figure()
                plt.imshow(dis.astype(np.uint8))
                plt.show()

            if self.pngsave:
                cv2.imwrite(self.kaggle_img_path + str(self.current_id+1) + "_img.png", crop_phase)
                cv2.imwrite(self.kaggle_mask_path + str(self.current_id+1) + "_mask.png", crop_binary)

            if self.dbsave:
                # add a row
                im_path = self.kaggle_img_path + str(self.current_id+1) + "_img.png"
                label_path = self.kaggle_mask_path + str(self.current_id+1) + "_mask.png"
                self.__update_to_db(label=label, time=i+1, features=features, im_path=im_path, label_path=label_path)
                self.current_id += 1

    def __label_analyzed(self, label_img):
        for label in range(2, 90):
            if len(label_img[label_img == label]) > 0:
                self.analysis_label.append(label)

    def __connect_to_db(self):
        """ MYSQL """
        passward = getenv("DBPASS")
        self.engine = create_engine('mysql+pymysql://BT:' + passward + '@127.0.0.1:3306/Cell')
        Session = sessionmaker(bind=self.engine, autoflush=False)
        self.sess = Session()
        print("Connect...")

    def check_last_id(self):
        assert self.sess is not None
        obj = self.sess.query(RetinalPigmentEpithelium).order_by(RetinalPigmentEpithelium.id.desc()).first()
        if obj is None:
            self.current_id = 0
        else:
            self.current_id = obj.id
        print("current id: ", self.current_id)

    def __update_to_db(self, label, time, features, im_path, label_path):
        id = self.current_id + 1
        year, month, day = self.date

        # check
        assert type(id) is int
        assert (type(year) is int) and (type(month) is int) and (type(day) is int)
        assert (type(label) is int) and (type(time) is int)
        assert len(im_path) < 150
        assert len(label_path) < 150
        assert type(features[0]) is float
        assert type(features[1]) is float
        assert type(features[2]) is float
        assert type(features[3]) is float
        assert type(features[4]) is bool
        assert type(features[5]) is float
        assert type(features[6]) is float

        # a row as a object
        tem = RetinalPigmentEpithelium(id, year, month, day, label, time, im_path, label_path, features)
        # add
        self.sess.add(tem)

    def __db_commit(self):
        self.sess.commit()
        self.engine.dispose()


class Fov(WorkFlow):
    def __init__(self, root):
        sys.stdout = open('file.txt', 'a+')
        super().__init__(root)
        file_number = len(glob.glob(self.root + "[0-9]*"))
        print("Found ", file_number, "pair image")
        self.file_list = [self.phase_npy_path + str(p) + "_phase.npy" for p in range(1, file_number+1)]
        self.pic_save = [self.pic_path + str(p) + ".png" for p in range(1, file_number+1)]
        self.cur_num = [str(p) for p in range(1, file_number+1)]
        self.__check_file_combo()
        sys.stdout.close()

    def __check_file_combo(self):
        for p in self.file_list:
            check_file_exist(p, p)

    def run(self):
        center = (IMAGESIZE//2, IMAGESIZE//2)
        radius = IMAGESIZE//2
        sys.stdout = open('file.txt', 'a+')
        for i, im_p, pic_p in zip(self.cur_num, self.file_list, self.pic_save):
            print(im_p)
            img = np.load(im_p)
            black = np.zeros((IMAGESIZE, IMAGESIZE), dtype=np.uint8)
            black = cv2.circle(black, center, radius, 1, thickness=-1)
            img[black == 0] = 0
            np.save(im_p, img)
            plt.figure(i)
            plt.title(i + "phase image")
            plt.imshow(img, cmap='jet', vmax=MAXPHASE, vmin=MINPHASE)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(pic_p)
            plt.close(i)
        sys.stdout.close()


