VALIS for ACROBAT 2023
======================

Chandler Gatenbee :sup:`1`, Alexander R.A. Anderson :sup:`1`

:sup:`1` Department of Integrated Mathematical Oncology, H. Lee Moffitt Cancer Center & Research Institute, 12902 Magnolia Drive, SRB 4, Tampa, Florida, 336122

VALIS :sup:`1` was used to perform registration of the test images provided as part of the ACROBAT 2023 grand challenge, although with a few updates. In particular, a different preprocessing method was used, and a second higher-resolution rigid registration was performed prior to non-rigid registration. These updates will be included as options in the next release of VALIS. The code used for the grand challenge, which includes these updates, can found on GitHub.

Preprocessing
*************

.. image:: https://github.com/MathOnco/valis/raw/main/examples/acrobat_2023/_images/processing_example.png

**Figure 1** Image pre-processing. a) Original moving image (top) and fixed image (bottom) b) Processed moving image (top) and processed fixed image (bottom).

The goal of this preprocessing method is to “flatten” the image, such that lightly stained regions get enhanced while heavily stained or very dark regions (often artifacts) get lightened. To accomplish this, the image is first converted to the polar CAM16-UCS colorspace to create a JCH image :sup:`2`. The JCH colors are then then clustered into 100 groups using K-means clustering, and the centroids of each cluster then converted back to RGB. The RGB centroids are used to create a “color matrix” for the image, which is in turn used to deconvolve the image into 100 channels using color deconvolution :sup:`3`. The mean for each pixel is calculated using all channels, creating a single channel image, which further undergoes adaptive histogram equalization. The results of the preprocessing can be seen in Figure 1.



High resolution rigid registration
**********************************

.. image:: https://github.com/MathOnco/valis/raw/main/examples/acrobat_2023/_images/mico_rigid_reg.png

**Figure 2** High resolution rigid registration. Left and right insets show which features were matched in the tiles extracted from the images aligned using preliminary rigid transforms (center). Fixed images are on the top, while the moving images are on the bottom.

The images used for the initial rigid registration are much lower resolution than the original WSI. While this allows VALIS to attain a good initial alignment, using features from the higher resolution may improve the rigid registration, potentially reducing unwanted/excessive non-rigid deformations, both of which should result in more accurate alignments. Higher resolution feature detection should also improve VALIS’ error estimates, which were not very accurate. As such, we have introduced a second rigid registration step that is performed on higher resolution images warped using the initial rigid transforms (1/8th the size of the full resolution registered WSI) (see Figure 2). The moving and fixed images are then divided into tiles (512 x 512 pixels), each of which is processed with the preprocessing method described above. After processing, each pair of tiles is normalized to one another using the approach described in the VALIS manuscript. Next, Super Point and Super Glue :sup:`4` are used to detect and match features in the moving and fixed tiles. After all tiles have gone through this process, all matched keypoints are combined and filtered using RANSAC and Tukey’s outlier approach (i.e. the same filtering steps conducted during the initial rigid registration). The high-resolution rigid transformation matrix, M', can then be estimated using these matched features. If it is found that this higher resolution rigid registration produced more matches and that the average distances between registered matched features is smaller than before, M' is kept. Otherwise, the rigid transformation matrix found using the lower resolution images is retained.

After the higher resolution rigid alignment is complete, registration proceeds as described in the VALIS manusript1, with the “micro-registration” being performed on images that are 0.25 of the full resolution WSI. After registration is complete, we estimate the error using only the rigid transform and using the rigid + non-rigid transform by calculating the distance between matched features using each approach. The provided landmarks are then warped using the transform (rigid or rigid + non-rigid) that had the lowest estimated error.


References
**********

1.	Gatenbee CD, et al. Virtual alignment of pathology image series for multi-gigapixel whole slide images. Nature Communications 14, 4502 (2023).

2.	Li C, et al. Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application 42, 703-718 (2017).

3.	Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol 23, 291-299 (2001).

4.	Sarlin P-E, DeTone D, Malisiewicz T, Rabinovich A. SuperGlue: Learning Feature Matching with Graph Neural Networks. In: CVPR) (2020).


