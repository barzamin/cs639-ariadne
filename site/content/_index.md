+++
+++

# <span class="sc">Ariadne</span>: a comparison of traditional CV and DNNs for PCB defect identification and localization

<p class="author">Erin Moon <code>&lt;limarshall@wisc.edu&gt;</code></p>

<main><article>

## Motivation
Surface defect identification is an important problem in many industrial processes, and is often accomplished through automated optical inspection (AOI).
In printed circuit board (PCB) manufacture, catching defects early in the production chain is extremely important; disconnected or shorted nets can be very difficult to diagnose or QA test in a final assembly. Automated optical inspection is thus widely deployed in the PCB industry, and is a topic of ongoing research in CV/ML; the latency and accuracy constraints are significant, since AOI operates as an online, interstitial part of the production line.

I am specifically interested in not just classifying, but *locating* defects; both methods thus focus on obtaining bounding boxes around each defect identified, and classifying those defects in an ontology useful to PCB lithographers. Go/no-go decisions on an entire board image are sufficient for immediate control of the assembly line, but data on fault location and type is useful long-term; for instance, one can produce heatmaps of fault locations, or identify specific lithomasking machines which need servicing based on fault type distribution.

Since PCB manufacturing is generally done via photomasking, our source masks form a perfect ground truth description of each of the target PCB's copper layers; this reduces the AOI problem to comparison between the "template" mask source image and a scan of the post-etching copper layer. Additionally, since circuit boards are generally extremely flat, dimensionally accurate, and formed of uniform material, the imaging environment can be heavily constrained to remove most obstacles to registration and comparison.

<figure>
<img src="img/windpup_top.png" style="max-width: 400px;">
<figcaption>
An example copper layer mask from a board design by the author; this is a <code>gerbview</code> visualization of one of the Gerber files which was sent to fab.
</figcaption>
</figure>

Given the above problem description, I chose to focus on comparison of observed copper layers with the source lithomasks. I implemented two fault detectors: a naive, morphology-based CV method and a (semi-)novel transfer learning method of my own design, and compared their performance on a real-world-like{% sidenote() %}Unfortunately, board fabrication houses generally operate as clients of other companies and do not own the IP they fabricate; as such, they cannot release production AOI datasets even if they wanted to.{% end %} dataset.

## Dataset
There is only one dataset of note for PCB trace inspection—[DeepPCB](https://github.com/tangsanli5201/DeepPCB)—but it is high quality, captured via linear CCD scanning, and pre-binarized; this is similar to how boards are actually imaged in factories. Since this dataset is high quality, and trace inspection is a relatively well-constrained problem, I chose to use it as the core of my project work.

<!-- The DeepPCB dataset contains 1500 pairs of (*reference image*, *image potentially containing defects*); defects are annotated with a bounding box and type. Since the dataset prealigns reference template and part image pairs, I can further augment the dataset by randomly perturbing the "measured" images for each reference image with affine transformations, to simulate the effects of rotation/translation/lifting during imaging. -->

<figure>
<div class="subfigs">
<img src="img/12100013_merged.png">
</div>
<figcaption>
An example image pair from the DeepPCB dataset. <b>Left</b>: annotated observed image; <b>right</b>: template image.
</figcaption>
</figure>

The dataset contains 1500 pairs of (*reference image*, *observed image potentially containing defects*); defects are annotated with a bounding box and type. The defect classes enumerated by the <!-- dataset --> authors are:
<!-- The DeepPCB dataset annotates defects with a bbox and type. The defect classes enumerated by the dataset authors are: -->

- `OPEN`: a full cut (region of missing copper) through a board trace
- `SHORT`: stray copper which electrically misconnects two board traces
- `MOUSEBITE`: a partial cut through a trace; the net is still connected, but tapers inappropriately at the defect
- `SPUR`: a protrusion from a copper region which does not result in a short
- `COPPER`: a misdeposition of copper disconnected from any nets, often a speck or a blob
- `PINHOLE`: a spot of missing copper within a larger copper region

The annotations are provided in an *ad-hoc* text form, with one bbox and defect type per line. I wrote a small adapter in my Python codebase to parse these forms and crawl the whole dataset, providing it through a simple iterator interface that I can either peruse manually for the CV method or shim to a `torch.utils.data.Dataset` for the DNN object detector.

## Methodology
### Classical CV
I am lucky to use a dataset which provides well-aligned pairs of clean binarized images as a starting point; due to low noise tolerance in traditional CV AOI methods, physically conditioning the imaging environment is extremely important, and that work has already been done for me. As such, I designed and implemented{% sidenote() %}All image processing is performed via <a href="https://scikit-image.org/"><code>scikit-image</code></a>.{% end %} a naive procedure to identify defect regions, given the pair of a reference template and observed board:

1. *XOR* the ground truth and observed images to produce a difference mask
2. Perform binary opening on this mask to eliminate spidery structures.
3. Perform binary opening on this mask to regenerate original boundaries.
4. Remove small holes.
5. Remove any straggler small connected objects.
6. Label the image via connected component analysis.

We then proceed to classify the labeled image. For each labeled defect, I
1. Discard defects with under a threshold number of pixels. Likely redundant given the above steps.
2. Compute the contour of the defect, via marching squares.
3. Compute white fractions (effectively, binary histogram) for the inner pixels and rim pixels of the defect in both the template and observed image.

<figure>
<img src="img/region0.svg">
<figcaption>
A defect identified from the xor defect map, and histograms of its body and edge pixel value distribution in the observed image.
</figcaption>
</figure>

This produces a feature vector; instead of manually picking heuristic thresholds, I train a one-versus-one SVM classifier{% sidenote() %}Via <a href="https://scikit-learn.org/stable/"><code>scikit-learn</code></a>'s <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"><code>SVC</code></a>.{% end %} with a radial basis kernel on the defect region features.
Intuitively, by computing the pixel distribution of an identified defect in both the template and observed image, we can determine whether the region contains erroneous copper, or is an erroneous void.

If featurization fails (more than one continuous contour), or a ground truth bbox does not overlap an identified bbox during training, the object is discarded. Likewise, when predicting, if a defect cannot be featurized, it is considered a classification error.

## Pairwise (Faster) R-CNNs: bifurcating the backbone


<figure>
<img src="img/pairwise_rcnn.png">
<figcaption>
The <strong>Pairwise Faster R-CNN</strong> architecture.
</figcaption>
</figure>

</article></main>