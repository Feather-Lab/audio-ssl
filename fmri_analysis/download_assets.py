"""
The following is a reduced version of `download_large_files.py` 
from the model_metamers_pytorch repository:
https://github.com/jenellefeather/model_metamers_pytorch/blob/f89cdad5c355081f97886863ed901fc9b34bce21/download_large_files.py#L36C1-L38C60

which was written support of the publication 
@article{feather2023model,
  title={Model metamers illuminate divergences between biological and artificial neural networks},
  author={Feather, Jenelle and Leclerc, Guillaume and M{\k{a}}dry, Aleksander and McDermott, Josh H},
  journal={Nature Neuroscience},
  year={2023},
}

The original code downloaded model weights and assets folder 
to suport model-fmri comparisons. This reduced version just 
downloads the assets folder. 

Original Author(s): Jenelle Feather
Current version by: Ian Griffith 
"""

import requests 
import tarfile
import sys
import os

ASSETS_LOCATION = 'assets/' 

def download_extract_remove(url, extract_location):
    
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

# Download the assets folder (366M)
url_assets_folder = 'https://mcdermottlab.mit.edu//jfeather/model_metamers_assets/pytorch_metamers_assets_folder.tar'
download_extract_remove(url_assets_folder, ASSETS_LOCATION)