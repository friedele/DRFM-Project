# DRFM
 <h1 align="center">
  <a href="https://github.com/dec0dOS/amazing-github-template">
    <img src="https://github.com/friedele/DRFM-Project/blob/Development/images/combined2_targets_RD.png" alt="Logo" width="325" height="325">
  </a>
</h1>

<div align="center">
  <br />
  <br />
  <a href="https://github.com/friedele/DRFM-Project/issues/new?assignees=&labels=bug&template=01_BUG_REPORT.md&title=bug%3A+">Report a Bug</a>
  ·
  <a href="https://github.com/friedele/DRFM-Project/issues/new?assignees=&labels=enhancement&template=02_FEATURE_REQUEST.md&title=feat%3A+">Request a Feature</a>
  .
  <a href="https://github.com/friedele/DRFM_Project/discussions">Ask a Question</a>
</div>

<div align="center">
<br />

[![license](https://img.shields.io/github/license/dec0dOS/amazing-github-template.svg?style=flat-square)](LICENSE)

[![PRs welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/dec0dOS/amazing-github-template/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
[![made with hearth by dec0dOS](https://img.shields.io/badge/made%20with%20%E2%99%A5%20by-dec0dOS-ff1414.svg?style=flat-square)](https://github.com/dec0dOS)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [Abstract](#abstract)
- [Problem Overview](#problem-overview)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## Abstract

<table>
<tr>
<td>

Radar electronic jammers are evolving from hostile nations thus becoming more complex and presenting serious issues when a radar system is trying to interrogate the actual targets of interest. Electronic jammers that present a serious challenge are in a class called Digital Radio Frequency Memory (DRFM).  DRFM techniques work by generating coherent false targets to the radar receiver based on an intercepted pulse signal from the radar transmitter. This will position false targets ahead or behind the radar target, thus masking the real target with false targets.  The false targets can also be manipulated in amplitude, phase, and frequency.

Traditional approaches to target detection and estimation for electronic jammers generally rely on parametric modeling, which can fail because it violates the strict assumptions of classical signal processing algorithms. The result is substantial algorithm performance degradation. Furthermore, parametric models to handle DRFM jammers are difficult to design and ineffective against an evolving DRFM technology.   The key to identifying opportunities for improved electronic jammer protection and signal processing in radars is to use machine learning techniques to challenge the underlying assumptions of the standard parametric approach to design and analyze radar systems

Convolutional Neural Networks (CNN) have gained popularity in the last few years with the advent of faster high-performance computer systems which rely on GPUs for the best computational performance.  A CNN operates from a mathematical perspective and is used for non-trivial tasks such as image classification. CNNs have great performance while classifying images when they are very similar to the training dataset. However, little work has been done in developing realistic radar models which are ignoring radar environmental and antenna effects thus providing inaccurate simulation training datasets and credibility.  In addition, current publicly available research does not typically consider the five dimensions of a radar sensor, thus presenting an incomplete signal processing chain. From a first principles perspective, the radar measures the following aspects of a signal target return:  Range, azimuth, elevation, Doppler, and signal amplitude.

We propose to design a CNN that will use spatial training datasets, where the radar signal processing concerning DRFM jamming will be examined to identify and classify DRFM-type jammers.  The CNN will use range-Doppler images as inputs to classify if our ground-based phased array radar system is being jammed by DRFM false targets. A suitable network architecture and values for hyperparameters are found by iterative experimental studies.  The correct classification of DRFM targets is what we desire while processing through the radar signal returns. The predictive quality of our CNN model will drive the radar system performance and support any further actions to mitigate a DRFM jammer attack.


<details open>
<summary>Additional info</summary>
<br>

This project is the result of ongoing research. 

</details>

</td>
</tr>
</table>

## Problem Overview
Ideally, radar signal processing should be able to accurately and efficiently detect false or misleading targets of interest.  The reality is that typical radar signal processing is ineffective against false targets produced by deceptive electronic jammers such as DRFM. Traditional approaches to target detection and track estimation for electronic countermeasures (ECM) in general, rely on parametric modeling, that can fail because it violates the strict assumptions of classical signal processing algorithms.  ECM, in general, attempts to interfere with or deceive the radar system with misleading electronic signals.  This failure occurs because the misleading signals and the real targets are processed in the same way as the output of the analog-to-digital converter, thus the false targets and real targets are mixed together.  The result is substantial algorithm performance degradation in both target detection and track performance.  The matched filter (see section 2.3.3) which optimizes the SNR for detection processing does not identify false targets.  The outcome of this shortcoming is that false targets are passed through to the point of detection processing (see Figure 1.1), which then can pollute the tracker and further task scheduling of radar resources towards the misleading signals. We argue that if we can identify the misleading signals, then the radar can take steps to mitigate or cancel the ECM. We also propose to identify that we have misleading signals before processing the detections, thus saving the radar from unnecessary computing cycles to process the false targets and prevent overall algorithm performance degradation. 

We propose to use machine learning techniques to challenge the underlying assumptions of the standard parametric approach for the design and analysis of radar systems.  Convolutional Neural Networks (CNN) have gained popularity in the last few years with the advent of faster high-performance Graphics Processing Unit (GPU) computers.  Current research demonstrates CNN as a sound approach for radar signal classification with more work to be done.  We will show a Convolutional Neural Network (CNN) that will use spatial training datasets in the form of range-Doppler images to perform radar signal classification.  The radar signal processing will then be examined to identify different ECM classes. The goal is to show that ECM can be mitigated and thus improve overall radar performance operation and target recognition. We focus on a particular type of active ECM called Digital Radio Frequency Memory (DRFM).  We choose to focus on the DRFM jammer because it’s the most problematic ECM for radars to date.

Success is defined by the identification of an ECM, in this case, a DRFM jammer, and its type via the CNN model.  The new CNN model is proposed to be part of the radar’s signal processing chain.  The DRFM jammer will fail to affect radar detection and tracking operations in the radar.  Identification is the first step in mitigating the effects of ECM.

The status quo is the failure to identify false targets or interference is what we face today since DRFM is problematic for typical signal processing.  The failure effects are as follows:

 - Impact on system performance due to more signal processing operations
 - Degrades detection performance 
 - Hides the real targets, deceiving the Tracker


## Roadmap

See the [open issues](https://github.com/friedele/DRFM-Project/issues) for a list of proposed features (and known issues).

- [Top Feature Requests](https://github.com/dec0dOS/amazing-github-template/issues?q=label%3Aenhancement+is%3Aopen+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Top Bugs](https://github.com/dec0dOS/amazing-github-template/issues?q=is%3Aissue+is%3Aopen+label%3Abug+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Newest Bugs](https://github.com/dec0dOS/amazing-github-template/issues?q=is%3Aopen+is%3Aissue+label%3Abug)

## Contributing


Please try to create bug reports that are:

- _Reproducible._ Include steps to reproduce the problem.
- _Specific._ Include as much detail as possible: which version, what environment, etc.
- _Unique._ Do not duplicate existing opened issues.
- _Scoped to a Single Bug._ One bug per report.

## Support

Reach out to the maintainer at one of the following places:

- [GitHub discussions](https://github.com/friedele/DRFM-Project/discussions)
- The email which is located [in GitHub profile](https://github.com/friedele)

## License

This project is licensed under the **MIT license**. Feel free to edit and distribute this template as you like.

See [LICENSE](LICENSE) for more information.

## Acknowledgements

Resources that were used during the development of the **DRFM-Project**:

- <https://pytorch.org>
- <https://pytorch.org/hub/pytorch_vision_resnet>
