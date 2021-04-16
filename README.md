# dpda-linux
Distribution-preserving data-augmentation 
DPDA v1.05
----------

1. Download a dataset:
 - UC Merced Land Use Dataset: http://weegee.vision.ucmerced.edu/datasets/landuse.html
 - Intel Image Classification: https://www.kaggle.com/puneet6060/intel-image-classification
 - The Oxford-IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
2. Install dependencies -- install_dependencies.sh (Minimum required OpenCV version is 4.51. You can install OpenCV 4.51 using the information at http://milq.github.io/install-opencv-ubuntu-debian/)
3. Go to the DPDA code directory
4. You must create two folders named "images" and "results" in "build" directory (They are provided)
5. Create a folder under "images" named "train"
6. Create as many folders for your classes as you wish under the "train" folder (Sample data is provided). 
7. Copy images to corresponding class folders -- png, jpg, and tiff are supported
8. Build and run using the following commands:
9. Go to "build" folder: 
   cd build
9. cmake ..
11. If everything ok:
   make -j 4 
   or 
   make -j 8 
   or 
   make -j 16  
   (change the number after j with the number of cpu cores for your system: 2-4-6-8-10-12-16-20)
10. Make the "DPPDA_Executable" as executable file:
    sudo chmod +x DPPDA_Executable 
11. Run executable:
   ./DPDA_Executable 
12. You will see the resulting augmented images under the results folder.
13. Augmentation settings can be adjusted by using the informative comments in the "main.cpp" file.
