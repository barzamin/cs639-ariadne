+++
+++

# <span class="sc">Ariadne</span>: a comparison of traditional CV and DNNs for PCB defect identification and localization

<p class="author">Erin Moon <code>&lt;limarshall@wisc.edu&gt;</code></p>

<main><article>

<h2>Motivation</h2>
Surface defect identification is an important problem in many industrial processes, and is often accomplished through automated optical inspection (AOI).
In printed circuit board (PCB) manufacture, catching defects early in the production chain is extremely important; disconnected or shorted nets can be very difficult to diagnose or QA test in a final assembly. Automated optical inspection is thus widely deployed in the PCB industry, and is a topic of ongoing research in CV/ML; the latency and accuracy constraints are significant, since AOI operates as an online, interstitial part of the production line.

In this project, I attempt to implement both a naive, morphology-based CV method and a (semi-)novel transfer learning method of my own design, and compare their performance in real-world scenarios. Thanks to the DeepPCB dataset (cite),

<h2>Dataset</h2>
The DeepPCB dataset annotates defects with a bbox and type. The defect classes enumerated by the dataset authors are:

- `OPEN`: a full cut (region of missing copper) through a board trace
- `SHORT`: stray copper which electrically misconnects two board traces
- `MOUSEBITE`: a partial cut through a trace; the net is still connected, but tapers inappropriately at the defect
- `SPUR`: a protrusion from a copper region which does not result in a short
- `COPPER`: a misdeposition of copper disconnected from any nets, often a speck or a blob
- `PINHOLE`: a spot of missing copper within a larger copper region

</article></main>