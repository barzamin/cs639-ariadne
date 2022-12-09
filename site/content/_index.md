+++
+++

# <span class="sc">Ariadne</span>: a comparison of traditional CV and DNNs for PCB defect identification and localization

<p class="author">Erin Moon <code>&lt;limarshall@wisc.edu&gt;</code></p>

<main><article>

<h2>Motivation</h2>
Surface defect identification is an important problem in many industrial processes, and is often accomplished through automated optical inspection (AOI).
In printed circuit board (PCB) manufacture, catching defects early in the production chain is extremely important; disconnected or shorted nets can be very difficult to diagnose or QA test in a final assembly. Automated optical inspection is thus widely deployed in the PCB industry, and is a topic of ongoing research in CV/ML; the latency and accuracy constraints are significant, since AOI operates as an online, interstitial part of the production line.

I am specifically interested in not just classifying, but *locating* defects; both methods thus focus on obtaining bboxes around each defect identified, and classifying those defects in an ontology useful to PCB lithographers. Go/no-go decisions on an entire board image are sufficient for immediate control of the assembly line, but data on fault location and type is useful long-term; for instance, one can produce heatmaps of fault locations, or identify specific lithomasking machines which need servicing based on fault type distribution.

Thus, in this project, I implemented two bounding-box fault predictor/classifiers: a naive, morphology-based CV method and a (semi-)novel transfer learning method of my own design, and compared their performance on a real-world-like{% sidenote() %}Unfortunately, board fabrication houses generally operate as clients of other companies and do not own the IP they fabricate; as such, they cannot release production AOI datasets even if they wanted to.{% end %} dataset.

<h2>Dataset</h2>
There is only one dataset of note for PCB trace inspection—<a href="https://github.com/tangsanli5201/DeepPCB">DeepPCB</a>—but it is high quality, captured via linear CCD scanning, and pre-binarized; this is similar to how boards are actually imaged in factories. Since this dataset is high quality, and trace inspection is a relatively well-constrained problem, I chose to use it as the core of my project work.

The DeepPCB dataset contains 1500 pairs of (*reference image*, *image potentially containing defects*); defects are annotated with a bounding box and type. Since the dataset prealigns reference template and part image pairs, I can further augment the dataset by randomly perturbing the "measured" images for each reference image with affine transformations, to simulate the effects of rotation/translation/lifting during imaging.

The DeepPCB dataset annotates defects with a bbox and type. The defect classes enumerated by the dataset authors are:

- `OPEN`: a full cut (region of missing copper) through a board trace
- `SHORT`: stray copper which electrically misconnects two board traces
- `MOUSEBITE`: a partial cut through a trace; the net is still connected, but tapers inappropriately at the defect
- `SPUR`: a protrusion from a copper region which does not result in a short
- `COPPER`: a misdeposition of copper disconnected from any nets, often a speck or a blob
- `PINHOLE`: a spot of missing copper within a larger copper region

The annotations are provided in an *ad-hoc* text form, with one bbox and defect type per line. I wrote a small adapter in my Python codebase to parse these forms and crawl the whole dataset, providing it through a simple iterator interface that I can either peruse manually for the CV method or shim to a `torch.utils.data.Dataset` for the DNN object detector.

<h2>Methodology</h2>
<h3>Classical(ish) Computer Vision</h3>
I am lucky to use a dataset which provides well-aligned pairs of clean binarized images as a starting point; due to low noise tolerance in traditional CV AOI methods, physically conditioning the imaging environment is extremely important, and that work has already been done for me. As such, I designed a fairly simple (naive, really!) procedure to identify defect regions, given the pair of a reference template and observed board:

1. *XOR* the ground truth and observed images to produce a difference mask
2. Perform binary opening on this mask to eliminate spidery structures.
3. Perform binary opening on this mask to regenerate original boundaries.
4. Remove small holes.
5. Remove any straggler small connected objects.
6. Label the image based on connectivity.

We then proceed to classify the labeled image. For each labeled defect, I
1. Discard defects with under a threshold number of pixels. Likely redundant given the above steps.
2. Compute the contour of the defect, via marching squares.
3. Compute white fractions (effectively, binary histogram) for the inner pixels and rim pixels of the defect in both the template and observed image.

**TODO: figures, etc**

This produces a feature vector; instead of manually picking heuristic thresholds, I train a one-versus-one SVM classifier with a radial basis kernel on the defect region features.

</article></main>