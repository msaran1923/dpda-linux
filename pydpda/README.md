# dpda-python
Distribution-preserving data-augmentation (Python)
DPDA v1.06p
----------

## Setup
1. Download a dataset and extract their contents into seperate class directories under "images/train" directory (samples are provided). Sample datasets are available at:
   - UC Merced Land Use Dataset: http://weegee.vision.ucmerced.edu/datasets/landuse.html
   - Intel Image Classification: https://www.kaggle.com/puneet6060/intel-image-classification
   - The Oxford-IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/

2. Install Python dependencies:

       pip install numpy pyflann3 h5py
   
   or
   
       pip install -r requirements.txt
   
   In addition, Python3 bindings for OpenCV (and OpenCV itself) must also be installed. For Debian/Ubuntu/etc:
   
       apt install python3-opencv
   
   Installation method for other distributions may vary, please check your distribution's own documentation.
   
## How To Use
Under "pydpda" directory, execute:

    python3 ./DPDA_Executable.py

You will see the resulting augmented images under the results directory. Augmentation settings can be adjusted by using the informative comments in the "DPDA_Executable.py" script.
