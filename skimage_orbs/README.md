# skimage-orbs-faiss-example

Using [Faiss](https://github.com/facebookresearch/faiss) to perform [ORB feature descriptor](https://en.wikipedia.org/wiki/ORB_%28feature_descriptor%29) matching with scikit-image.

This example is adapted from the scikit-image example for the [ORB feature detector and binary descriptors](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html), and could be adapted for other binary descriptors.

It takes two image filenames as arguments, computes ORB feature descriptors for each, uses FAISS to find cross-checked matches, and plots the results.

See the [Faiss documentation on Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes) if you'd like to switch to using a faster index like `IndexBinaryIVF`.
