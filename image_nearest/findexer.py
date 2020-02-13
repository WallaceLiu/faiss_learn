import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import binascii
import base64
import faiss
from faissconfig import *


class IndexerTrain(object):
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create(
            nfeatures=NUM_FEATURES,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
        )
        self.dimensions = DIMENSIONS

    def calc_sift(self, image):
        if not os.path.isfile(image):
            print("Image: does not exist")
            return -1, None

        try:
            image_o = cv.imread(image)
        except:
            print("Open Image: failed")
            return -1, None

        if image_o is None:
            print("Open Image:{} failed".format(image))
            return -1, None

        image = cv.resize(image_o, (NOR_X, NOR_Y))
        if image.ndim == 2:
            gray_image = image
        else:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        kp, des = self.sift.detectAndCompute(gray_image, None)

        sift_feature = np.matrix(des)
        return 0, sift_feature

    def iterate_files(self):
        print("iterating images from path: ", TRAIN_IMAGE_DIR)
        result = []
        for root, dirs, files in os.walk(TRAIN_IMAGE_DIR, topdown=True):
            for fl in files:
                if fl.endswith("jpg") or fl.endswith("JPG"):
                    result.append(os.path.join(root, fl))
        return result

    def indexer(self):

        index = faiss.index_factory(self.dimensions, INDEX_KEY)
        if USE_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        images_list = self.iterate_files()
        # prepare ids
        ids_count = 0
        index_dict = {}
        ids = None
        features = np.matrix([])
        for file_name in images_list:
            ret, sift_feature = self.calc_sift(file_name)

            if ret == 0 and sift_feature.any():
                # record id and path
                image_dict = {ids_count: (file_name, sift_feature)}
                index_dict.update(image_dict)
                ids_list = np.linspace(
                    ids_count, ids_count, num=sift_feature.shape[0], dtype="int64"
                )
                ids_count += 1
                if features.any():
                    features = np.vstack((features, sift_feature))
                    ids = np.hstack((ids, ids_list))
                else:
                    features = sift_feature
                    ids = ids_list
                if ids_count % 500 == 499:
                    if not index.is_trained and INDEX_KEY != "IDMap,Flat":
                        index.train(features)
                    index.add_with_ids(features, ids)
                    ids = None
                    features = np.matrix([])

        if features.any():
            print("training..")
            if not index.is_trained and INDEX_KEY != "IDMap,Flat":
                index.train(features)
            index.add_with_ids(features, ids)

        # save index
        print("saving index..")
        faiss.write_index(index, INDEX_PATH)
        # save ids
        with open(IDS_VECTORS_PATH, "wb+") as f:
            try:
                pickle.dump(index_dict, f, True)
            except EnvironmentError as e:
                print("Failed to save index file error:[{}]".format(e))
                f.close()
            except RuntimeError as v:
                print("Failed to save index file error:[{}]".format(v))
        f.close()
        # print("N", index.ntotal, dir(index), index.__dict__)
        return index.ntotal
