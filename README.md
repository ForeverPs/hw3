##Introduction
This is a simple programme for digital image processing, so what can we do?
* Get Histogram of input image(.bmp)
* Perform Histogram Equalization
* Perform Local Histogram Equalization  **just a little slow**
* Perform Histogram Matching 
* Perform Image Segmentation
* Draw Picture of an Image list with its corresponding names
##Samples
**1.** If I want to get the distribution and histogram values of a picture, I can do it like this:
* num, freq = histogram(img)

***I have to denote that img there is an array or matrix, num is a dict while freq is a list***

**2.** If I want to draw the histogram of input image, I can do it like this:
* num, freq = histogram(img) 
* draw(['Histogram'], [num], row=1, col=1, c='blue')

**3.** If I want to perform histogram equalization, I can do it like this:
* d, orig, _d, new = equal(filename, s_flag=False)
* namelist = ['Original histogram','Original image','Histogram after equalization','Equalization image']
* imglist = [d, orig, _d, new]
* draw(namelist, imglist, row=2, col=2, c='blue')

***Attention: d and _d are dict, orig and new are array,filename is the path of image***

**4.** If I want to perform local histogram equalization, I can do it in this way:
* size = [3, 3]
* step = 3
* orig, new = local_histogram(filename, size, step)
* draw(['orig', 'local equalization'], [orig, new], row=1, col=2)

**5.** If I want to perform histogram in my way, we call this histogram matching, we can do it in this way:
* orig, new = histogram_match(filename, pz)
* draw(['orig', 'after'], [orig, new], row=1, col=2)

***Denote:pz here is a list contains frequency of each gray value. If the frequency of one gray value is zero, you can not delete it, for example:[0.1, 0, 0, 0.2, 0.4, 0.3, 0, 0] stands for L = 8, if the max gray value is 255, the length of pz will be 256(contain 0)***

**6.** If I want to perform image segmentation, I can do it like this:
* img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
* background, object = segment(img)
* draw(['back', 'object'], [background, object], row=1, col=1)

