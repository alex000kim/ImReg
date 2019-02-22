# ImReg
### Image Registration Algorithms

Some domains (remotely sensing, medical imaging, etc) require subpixel level accuracy of image registration.
Here you can find the following algorithms:
- `mi_reg.py` Mutual Information (MI)

Zhang, Boyang, et al. "A mutual information based sub-pixel registration method for image super resolution." Intelligent Information Hiding and Multimedia Signal Processing, 2009. IIH-MSP'09. Fifth International Conference on. IEEE, 2009.

- `poc_reg.py` Phase-Only Correlation (POC)

Nagashima, Sei, et al. "A subpixel image matching technique using phase-only correlation." Intelligent Signal Processing and Communications, 2006. ISPACS'06. International Symposium on. IEEE, 2006.

Rule of thumb:
- `MI` is slow, but better at registering multi-modal images (i.e. X-ray vs CT scan vs MRI)
- `POC` is fast, but might have issues if images have significant differences 

--------------------

Original image

![alt image](image_original.jpg "Original image")


Shifted image (5.3p in X, 6.7p in Y)

![alt image](image_shifted.jpg "Shifted image")


```python
from mi_reg import main_mi_reg
from poc_reg import main_poc_reg

dx, dy, match_height = main_mi_reg(im_orig, im_shifted)
print(dx, dy, match_height)
> -5.300313983855842 -6.699973244887778 0.9967009221534583

dx, dy, match_height = main_poc_reg(im_orig, im_shifted)
print(dx, dy, match_height)
> -5.299159785340295 -6.700930722501576 0.9797807076193092


```
