import skimage as ski
import skimage.morphology as morph
import numpy as np
from .util.deeppcb import Defect
from typing import List

# img_pair : (img_truth, img_obsv)

class Unclassifiable(Exception): ...

def bbox_overlaps_defect(bbox, defect):
    minr, minc, maxr, maxc = bbox

    return not (minc > defect.x1 or minr > defect.y1 or maxc < defect.x0 or maxr < defect.y0)

def get_defect_blobs(img_pair, closing_structure=None):
    img_truth, img_obsv = img_pair
    closing_structure = closing_structure or morph.square(3)

    defect_mask = morph.binary_opening(img_truth ^ img_obsv)
    blobs = morph.binary_closing(defect_mask, closing_structure)
    blobs = morph.remove_small_holes(blobs)
    blobs = morph.remove_small_objects(blobs)
    labels = ski.measure.label(blobs)

    return (blobs, labels, defect_mask)

# wf -> "white fraction"
def _wf(x):
    return np.count_nonzero(x)/x.size

# def featurize(img_pair, ground_truth: List[Defect] = None, blob_thresh=10):
#     img_truth, img_obsv = img_pair
#     blobs, labels, defect_mask = get_defect_blobs(img_pair)

#     regiondata = []

#     for region in ski.measure.regionprops(labels):
#         if region.area < blob_thresh: continue

#         defect = None
#         if ground_truth: # training & have ground truth
#             matches = [x for x in ground_truth if bbox_overlaps_defect(region.bbox, x)]
#             if len(matches) == 0:
#                 continue # train should just skip stuff we can't figure out
#             defect = matches[0]

        # contours = ski.measure.find_contours(labels == region.label, 0, fully_connected='high')
        # if len(contours) != 1:
        #     raise Unclassifiable()

#         contour = contours[0]

#         # get the image coordinates of the contour
#         rows, cols = contours[0].astype(np.int64).T

#         # measure white fraction within defect blob and on contour, for both template and observed
#         features = np.array([
#             _wf(img_obsv[rows, cols]),
#             _wf(img_obsv[labels==region.label]),
#             _wf(img_truth[rows, cols]),
#             _wf(img_truth[labels==region.label]),
#         ])

#         regiondata.append({'defect': defect, 'bbox': region.bbox, 'features': features})

#     return regiondata

def featurize_region(img_pair, labels, region):
    img_truth, img_obsv = img_pair

    contours = ski.measure.find_contours(labels == region.label, 0, fully_connected='high')
    if len(contours) != 1:
        return None
    contour = contours[0]

    # get the image coordinates of the contour
    rows, cols = contour.astype(np.int64).T
    # measure white fraction within defect blob and on contour, for both template and observed
    features = np.array([
        _wf(img_obsv[rows, cols]),
        _wf(img_obsv[labels==region.label]),
        _wf(img_truth[rows, cols]),
        _wf(img_truth[labels==region.label]),
        region.eccentricity,
    ])

    return features
