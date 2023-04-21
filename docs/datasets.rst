Datasets
********

This section is intended to provide information about datasets related to WSI registration and benchmarking.

ANHIR
=====
The `Automatic Non-rigid Histological Image Registration (ANHIR) <https://anhir.grand-challenge.org>`_ was part of 2019 the IEEE International Symposium on Biomedical Imaging (ISBI). The aim of the challenge is to automatically perform non-rigid registraiton of 481 brightfield WSI pairs stained with different dyes. The high resolution WSI (up to 40x) came from 49 unique smamples (multiple images per sample) that a variety of source tissues, including lung lesions, lung lobes, mammary glands, colon adenoma carcinomas (COAD), mice kidneys, gastric mucosa, breast tissue, and kidney tissue. Hand annotated landmarks are provided for each "moving" image, with corresponding landmarks provided for 230 "fixed" images. These 230 images can therefore be used to train/tune the algorithm. The remaining 251 fixed landmarks are not available, but are used to caculate regisgtration accuracy when the results are submitted. Registration accuracy is primarily measured by the median relative target regisgtration error (rTRE), which is the distance between registrated landmarks (i.e. warped moving landmarks and fixed landmarks), divided by the fixed image's diagonal, thus providing a normalized measure of accuracy.


Benchmarking indicates that VALIS performs competitively using the defuault parameters, being one of the more accuracte opensource methods. Below is a figure showing the rTRE scores across a range of image sizes used for registration.


.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/anhir_results.png

This next plot shows the effect of increasing the size of the image used in the micro-registration, where the y-axis is the difference in landmark distance compared to the results when not using micro-registration (i.e. only using the lower resolution image to perform the registration). Interestingly, increasing the image size used to perform registration did not always increase accuracy, and when there were improvements, they were not always substantial.


.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/anhir_payoff_h.png


ACROBAT
=======

The `AutomatiC Registration Of Breast cAncer Tissue (ACROBAT) <https://acrobat.grand-challenge.org>`_ was part of MICCAI 2022 and, similar to ANHIR, the goal is to automatically register pairs of WSI. However, unlike ANHIR, the images were collected from routine diagonstic workflows, and so often contained artifcats common to real world datasets, such as cracks, streaks, pen marks, bubbles, etc... that incraese the difficulty of image registraiton. There were are total of 400 unique samples used to asses registration accuracy, with 100 images pairs used for a validation leaderboard (used to aid in developing the algorithm), and 300 for a test leaderboard. Unlike ANHIR, fixed landmarks were not provided, meaning that it is not possible to train/tune the method using matched landmarks. Scores can only be calculated by upload the registrated landmarks to the submission system. The primary score used to measure accuracy was the 90th percentile of physical distances between registered moving and fixed landmarks, in μm.

Due to the challegning nature of the images, a custom :code:`ImageProcesser` class was created to clean up and process the images. All image pairs webutre then registered using the defaults, followed by micro-registration using an image that was 25% of the full image's size. The script used to create this :code:`ImageProcesser` and conduct the registgration can be found `here <https://github.com/MathOnco/valis/blob/main/examples/acrobat_grand_challenge.py>`_. Using this approach, VALIS placed second overall, and first among the opensource methods.

As ACROBAT measures error in μm, and VALIS estimates error based on matched features, this dataset makes it possible to determine how well VALIS' error estimates match up with reality. The plot below shows the relationship between the estimated (VALIS) and true (ACROBAT) errors, with VALIS estimated error on the x-axis, error based on ACROBAT's hand annotations on the y-axis, and the identity line in red. These results indicate that VALIS tends to overestimate the error, with the actual accuracy being much better. This discrepancy may be due to the fact that the features used by VALIS to estimate error are based on much smaller versions of the images, and so their position is not as precise as those detected by hand.


.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/acrobat_error_comparison.png

Kartasalo et al 2018
=====================

Another potential use of image registration is to construct a 3D tissue from serial slices. In 2018, `Kartasalo et al. (2018) <https://academic.oup.com/bioinformatics/article/34/17/3013/4978049>`_ perfomed a study in which they compared several different frameworks for constructing 3D images, using both free and commercial software. They peformed the analysis using two datasets: one murine prostate to be reconstructed from 260 serially sliced 20x H&E images (0.46μm/pixel, 5μm in depth), and one murine liver to be reconstructed from 47 serial slices (0.46μm/pixel, 5μm in depth). Accuracy of the alignment of the liver can be determined using the positions of laser cut holes that pass through the whole tissue, and should in theory form a straight line. In the case of the prostate, for each pair of images, human operators determined the location of structures visible on both slices, preferably nuclei split by the sectioning blade. The authors refer to these landmarks as "fiducial points".

VALIS was used to register both datasets, and error was measured as the physical distance (μm), i.e. TRE, between the fiducial points. These values can then be comapred to those presented in Tables 1 and 2 of the manuscript, which provide the mean TRE using observer 1's landmarks (i.e. the "TRE μ" column). Benchmarking using the liver dataset indcates that VALIS produces a mean TRE of 52.98, compared to the compared to the baseline reference value of 27.3 (LS 1). In the case of prostate, VALIS scored 11.41, compared to the baseline reference value of 15.6 (LS 1). According to the authors, methods with scores apporaching the LS 1 value can be considered "highly accurate", indicating that VALIS is suitable for 3D reconstruction. Below is a picture of the prostate tumor reconstructed from all 260 serial slices.

.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/3d_recon.png


Similar to ACROBAT, this dataset provides the opportuinty to compare VALIS' error estimate to those based on manual measurements (e.g. the fiducial points). For the same reasons as before, it appears that VALIS over-estimates the error, as shown in the plots below.


.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/3D_error_plot.png
