# rainbow-umbrella
facial expression recognition in the presence of masks. 

## Required Python Libraries:
numpy torch matplotlib opencv-contrib-python scikit-learn

## Image database
This project is done using the chicago database, which you can download from [here](https://chicagofaces.org/default/)

## Instructions to run code:
### nmf.py, occlusion.py:
if running demo, add files to cwd and update filenames in main
if running on db, update config.py with source and destination folders (with absolute path)

### gabor.py
variable PATH: change the relative path to point to the folder containing the folder
variable output_folder_path: change the relative path to point to the desired destination location for the output for gabor filter

### ml.py
Store the images in '/gabor_output_2/gabors' in the current working directory and run the file

### nmf_ml.py
Store the images in '/nmf_dest_occluded' in the current working directory and run the file