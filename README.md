# dpda-linux
Distribution-preserving data-augmentation 
DPDA v1.05
----------

1. Install dependencies -- install_dependencies.sh (If you face with OpenCV problems, install OpenCV using the information at http://milq.github.io/install-opencv-ubuntu-debian/)
2. Go to the DPDA code directory
3. You must create two folders named "images" and "results" in "build" directory (They are provided)
3. Create a folder under "images" named "train"
4. Create as many folders for your classes as you wish under the "train" folder (Sample data is provided). 
5. Copy images to corresponding class folders -- png, jpg, and tiff are supported
6. Build and run using the following commands:
7. Go to "build" folder: 
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
