# facial_expression
## Installation
* A CUDA-enabled GPU is **strongly** recommened. 
* Manul Installation guidance is provided as follows. If a version number is specified, please follow it to ensure no error occurs. 
```bash
pip install opencv-python
pip install scikit-learn==0.22.2.post1
pip install scikit-image
pip install torch
pip install torchvision
pip install face-alignment
pip install tqdm
pip install numpy
pip install pandas
```
* The author used python 3.7. Other versions are not tested.
*  After these installations, you should be able to run webcam.py and webcam_scn.py. To run other codes, other installations might be needed.

## Usage
*  The current code configuration is for windows system. If running on other platforms, minor changes might be needed. 
* Download [svm_linear.pkl] and [epoch46_acc0.8703.pth]. Put them under ./model. The first is reqired by webcam.py and the latter by webcam_scn.py.
* To run the read-time facial expression classification implemented by SVM, run the following commands:
```bash
python3 webcam.py
```
* To run the read-time facial expression classification implemented by [Self-Cure-Network], run the following commands:
```bash
python3 webcam_scn.py
```

[svm_linear.pkl]: https://drive.google.com/file/d/168ybP_IQ_Hz7vYPZCVva4V3SUfTswmtm/view?usp=sharing
[epoch46_acc0.8703.pth]:
https://drive.google.com/file/d/1kkyWX4JJUkZCOwEbxn7-IpSE9lQTXfT9/view?usp=sharing
[Self-Cure-Network]:
https://github.com/kaiwang960112/Self-Cure-Network
