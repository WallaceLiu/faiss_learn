import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import heapq
from utils import read_array
import faiss
import binascii
import base64
from faissconfig import *


class FaissSearch(object):
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        self.sift = cv.xfeatures2d.SIFT_create()
        self.index_dict = self.get_vector_ids()
        self.results = []

    def isBase64(self, sb):
        try:
            if isinstance(sb, str):
                sb_bytes = bytes(sb, "ascii")
            elif isinstance(sb, bytes):
                sb_bytes = sb
            else:
                raise ValueError("Argument must be string or bytes")
            return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        except Exception:
            return False

    def extract_features(self, image):

        if self.isBase64(image):
            img_b64 = base64.b64decode(image)
            image = np.fromstring(img_b64, dtype=np.uint8)
            image = cv.imdecode(image, 1)

        else:
            image = cv.imread(image)
        image = cv.resize(image, (NOR_X, NOR_Y))
        if image.ndim == 2:
            gray_image = image
        else:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        kp, des = self.sift.detectAndCompute(gray_image, None)

        sift_feature = np.matrix(des)
        return 0, sift_feature

    def get_vector_ids(self):
        if not os.path.exists(IDS_VECTORS_PATH):
            return None

        with open(IDS_VECTORS_PATH, "rb") as f:
            index_dict = pickle.load(f)
        return index_dict

    def id_to_vector(self, id_):
        try:
            return self.index_dict[id_]
        except:
            pass
        raise Exception("no index dict found")

    def search_by_vectors(self, vectors):
        vectors = read_array(vectors, SIFT_DIMENSIONS)
        # ====== trick code start ===========
        count = vectors.shape[0]
        vectors = np.vstack((vectors, vectors))
        vectors = vectors[0:count, :]
        # ====== trick code end ===========
        ids = [None]
        results = self.search(ids, [vectors])
        return results

    def search_by_image(self, image_path):
        ret, vectors = self.extract_features(image_path)
        return self.search([None], vectors)

    def search(self, ids, vectors):
        def result_dict_str(id_, neighbors):
            return {"id": id_, "neighbors": neighbors}

        def neighbor_dict_with_path(id_, file_path, score):
            return {"id": int(id_), "file_path": file_path, "score": score}

        def neighbor_dict(id_, score):
            return {"id": int(id_), "score": score}

        scores, neighbors = self.index.search(vectors, k=TOP_N)
        n, d = neighbors.shape
        result_dict = {}
        for i in range(n):
            l = np.unique(neighbors[i]).tolist()
            for r_id in l:
                if r_id != -1:
                    score = result_dict.get(r_id, 0)
                    score += 1
                    result_dict[r_id] = score

        h = []
        need_hit = SIMILARITY
        for k in result_dict:
            v = result_dict[k]
            if v >= need_hit:
                if len(h) < TOP_N:
                    heapq.heappush(h, (v, k))
                else:
                    heapq.heappushpop(h, (v, k))
        result_list = heapq.nlargest(TOP_N, h, key=lambda x: x[0])

        neighbors_scores = []

        for e in result_list:
            confidence = e[0] * 100 / n
            if self.id_to_vector:
                file_path = self.id_to_vector(e[1])[0]
                neighbors_scores.append(
                    neighbor_dict_with_path(e[1], file_path, str(confidence))
                )
            else:
                neighbors_scores.append(neighbor_dict(e[1], str(confidence)))

        self.results.append(result_dict_str([None], neighbors_scores))
        if len(self.results) and len(self.results[0].get("neighbors")):
            return self.results[0].get("neighbors")
        return []
