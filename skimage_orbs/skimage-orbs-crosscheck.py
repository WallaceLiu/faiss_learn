#!/usr/bin/env python3

from skimage import data
from skimage import io
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
import pprint
import faiss
import numpy

# converts an array of ORB descriptors into the binary format expected by Faiss
def orb_descriptors_to_faiss( input_descriptors ):
    faiss_descriptors = []
    for descriptor in input_descriptors.astype(int):
        descriptor_string = ''.join(map(str,descriptor))
        descriptor_as_bytes = int(descriptor_string, 2).to_bytes(len(descriptor_string) // 8, byteorder='big')
        faiss_descriptors.append([descriptor_as_bytes[i] for i in range (0, len(descriptor_as_bytes))])
    return numpy.array(faiss_descriptors).astype('uint8')

descriptor_length = 256

img1 = io.imread(sys.argv[1], True)
img2 = io.imread(sys.argv[2], True)

descriptor_extractor = ORB(n_keypoints=200)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

# Original skimage matching code, uncomment to validate Faiss results:
# matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
# pprint.PrettyPrinter(indent=4).pprint(matches12)

index1 = faiss.IndexBinaryFlat(descriptor_length)
faiss_descriptors1 = orb_descriptors_to_faiss(descriptors1)
index1.add(faiss_descriptors1)

index2 = faiss.IndexBinaryFlat(descriptor_length)
faiss_descriptors2 = orb_descriptors_to_faiss(descriptors2)
index2.add(faiss_descriptors2)

D1, I1 = index1.search(faiss_descriptors2,1)
D2, I2 = index2.search(faiss_descriptors1,1)

# store cross-checked matches in faiss_matches
# i.e. a matched pair (keypoint1, keypoint2) is returned if keypoint2 is the
# best match for keypoint1 in second image and keypoint1 is the best
# match for keypoint2 in first image.
faiss_matches = []
for index, value in enumerate(I2):
    if I1[value[0]] == index:
        faiss_matches.append([index, value[0]])
faiss_matches = numpy.array(faiss_matches)
# faiss_matches should now be the same as matches12 above, uncomment to validate:
# pprint.PrettyPrinter(indent=4).pprint(faiss_matches)
# print(numpy.array_equal(matches12,faiss_matches))

fig, ax = plt.subplots(nrows=1, ncols=1)

plt.gray()

plot_matches(ax, img1, img2, keypoints1, keypoints2, faiss_matches)
ax.axis('off')
ax.set_title("Original Image vs. Transformed Image")

plt.show()
