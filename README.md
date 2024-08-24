![Header](images/colorbanner.jpg)
# Color-Based Image Retrieval Algorithm and Color Histogram Generator

## Description
This project is a color-based image retrieval algorithm that allows users to retrieve images from a database based on their color similarity to a query image. The algorithm can extract color histograms in the color space of RGB, HSV, or CIELAB and display them. It utilizes the equations of euclidean, cosine, chi2, correlation, intersection, and bhattacharyya to compare and rank color histograms. it contains a function that helps optimize the retrieval process for large datasets.

## Background & Potential Uses
This algorithm is a simplified version of a project I worked on to help law enforcement retrieve images of indoor spaces and geolocate a query image. However, a color-based image retrieval algorithm is also useful for graphic design, shopping, art curating, and any field where color characteristics play an important role in image analytics. 

## Output Examples
**Color-Based Image Retrieval:**
<p align="center">
    <img src="images/retrievalExamp.png" width="700" height="150">
</p>

**Color Histogram Extraction:**
<p align="center">
<img src="images/colorHistRGB.png" width="200" height="150">
<img src="images/colorHistHSV.png" width="200" height="150">
<img src="images/colorHistCIELAB.png" width="200" height="150">
</p>

## Usage
For the usage instructions please look at the ['CodeDemo.py'](CodeDemo.py) file. Additionally, look at the ['ColorCBIR.py'](ColorCBIR.py) file for the code source and explanation of functions.
<p align="center">
    <img src="images/codeImage.png" width="600" height="195">
</p>
