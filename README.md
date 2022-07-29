

<div align="center">
  <h1>Motorcycle Rods Inspection</h1>
  
  <p>
    Machine-Vision Application
  </p>

  
<!-- Badges -->
<p>
  <a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Louis3797/awesome-readme-template" />
  </a>
  <a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/network/members">
    <img src="https://img.shields.io/github/forks/CDOrtona/Rod_Inspection_Machine_Vision" alt="forks" />
  </a>
  <a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/stargazers">
    <img src="https://img.shields.io/github/stars/CDOrtona/Rod_Inspection_Machine_Vision" alt="stars" />
  </a>
  <a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/CDOrtona/Rod_Inspection_Machine_Vision" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/Report.pdf">Documentation</a>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Features Extraction](#features-extraction)
  * [Nuisance Handling](#nuisance-handling)
- [Machine Vision Workflow](#project-workflow)
- [Usage](#usage)
  * [Dependencies](#dependencies)
  * [Set-Up](#set-up)
- [Contributors](#contributors)
- [License](#license)
  

<!-- About the Project -->
## About the Project


The developed software system aimed at visual inspection of motorcycle 
connecting rods. The system is able to analyse the dimensions of two different types
of connecting rods to allow a vision-guided robot to pick and sort rods based on their type and 
dimensions. The two rod types are characterized by a different number of holes: Type A rods have 
one hole whilst Type B rods have two holes.
Below it has been illustrated a few examples of the pictures which the software has been programmed to analyze;
the full list can be foud [here](https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/tree/master/images).

<p float="left">
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/images/TESI12.BMP" width="" />
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/images/TESI50.BMP" width="" /> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/images/TESI92.BMP" />
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/images/TESI49.BMP" />
</p>

<!-- First Task -->
### Features Extraction
The developed software is capable of acquiring the following features from each image:
- Number of Rods
- Type of Rod (Type A or Type B)
- Position and Orientation
- Length, Width, Width at the barycenter
- For each hole: position of the barycenter and diameter

<!-- Second Task -->
### Nuisance Handling 
The program is robust to the following possible issues one might run into when analyzing the rods:
- Iron Powder hindering segmentation
- Rods touching with one another
- Images might contain objects which are not rods and should not be further analyzed



<!-- Workflow -->
## Project Workflow

<div align="center"> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/task2wf.png" alt="screenshot" />
</div>

The picture above depicts the whole project's workflow.

The first step consists in the foregorund/background segmentation of the image, which has been implemented with the [Otsu's algorithm](https://en.wikipedia.org/wiki/Otsu%27s_method).
This was possible because the images have been acquired through the so called backlighting technique, hence their histogram appear inherently binary.
However, some of the images are affected by the presence of iron powder, which must be smoothed out before the segmentation phase otherwise they might be wrongly
classified as foreground pixels. This issue has been tackled through the implementation of a median filter which smooths the image from the iron powder which is treated as
impulse noise.

<div align="center"> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/task2iron.png"/>
</div>

Once the image has been correctly segmentated every element belonging to the foreground must be told apart with the aid of the Connected-Component Labeling.
Before this can be performed, however, every touching rod must be separated otherwise two separate rods might be labeled as belonging to one single rod.
In order to detach touching components each component have been enclosed with a convex hull and then the points where the rods are touching are 
computed through the convexity defect function. 

Below it is demonstrated how two touching rods are separated.

<div align="center"> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/detatch.png"/>
</div>

Each rod is assigned a label in the Connected-Component labeling and a RGB-mask is applied to highlight them.

<div align="center"> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/mask_rgb.png"/>
</div>

Before performing the Blob Analysis where the features of each Blob is computed, those elements present in the pictures which are not rods(washers and screws) are discarded.

<div align="center"> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/mask_rgb.png"/>
</div>

The results of the blob analysis are depicted below:

<p float="left">
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/Figure_1.png" width="300" />
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/Figure_2.png" width="300" /> 
  <img src="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/Figure_3.png" width="300"/>
</p>



The results are also printed on screen and saved in an excel file.


```bash
  [ Rod type: A 
num holes: 1 	 -> holes: [{'D': 24.70020388546793, 'Cx': 45.06264775413712, 'Cy': 115.52009456264776}] 
ib: 82.35443425076453, jb: 82.12721712538226 
orientation: 47.951162391517 8barycenter width: 16.278820596099706 
width: 44.72901342652703 	 length: 151.86274093262938 
 
,  Rod type: B 
num holes: 2 	 -> holes: [{'D': 23.799887584719077, 'Cx': 86.9047619047619, 'Cy': 119.10273368606701}, {'D': 26.68729810895878, 'Cx': 163.10987529491067, 'Cy': 114.71149309066396}] 
ib: 128.38539553752537, jb: 116.62407031778228 
orientation: 86.69539973760394 barycenter width: 17.029386365926403 
width: 35.345310989417506 	 length: 100.46780517663012 
 
,  Rod type: A 
num holes: 1 	 -> holes: [{'D': 25.150362035842353, 'Cx': 149.78933333333333, 'Cy': 167.61904761904762}] 
ib: 104.70156106519742, jb: 172.1900826446281 
orientation: 84.62185250686889 barycenter width: 19.1049731745428 
width: 50.06107504951664 	 length: 136.99468496705683 
 
]
```

Check the [documentation](https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/blob/master/project_files/Report.pdf) for more in-depth information.






<!-- Usage -->
## Usage

### Dependencies

- [Python 3.9](https://www.python.org/downloads/)
- [OpenCV](https://docs.opencv.org/4.x/index.html)
- [MatplotLib](https://matplotlib.org/)
- [Pandas]()


### Set-Up

Clone the project

```bash
  git clone https://github.com/CDOrtona/Rod_Inspection_Machine_Vision.git
```

Go to the project directory

```bash
  cd Rod_Inspection_Machine_Vision
```

Install the required python modules:

```bash
  pip install -r requirements.txt
```

<!-- Contributors -->
## Contributors

<a href="https://github.com/CDOrtona/Rod_Inspection_Machine_Vision/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CDOrtona/Rod_Inspection_Machine_Vision" />
</a>

<!-- License -->
## License

Distributed under the MIT License. See LICENSE.txt for more information.
