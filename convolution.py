# Script for Using different filters for applying convolutions

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.exposure import rescale_intensity

# function for applying custom convolution
def convolution(image, filter):
    '''
    Parameters:
    image : input image matrix (numpy array), gray scaled
    filter : kernel which is to be applied using sliding window approach

    Returns:
    output : output image (numpy array)
    '''
    # getting the image height and width
    ih, iw = image.shape[0], image.shape[1]
    # getting the kernel height and width
    kh, kw = filter.shape[0], filter.shape[1]
    # calculating the padding required to make output image size same as input image size
    pad = (kw - 1) // 2
    # padding using replicating border pixels , pad value same on top, bottom, left and right
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # output image will be same as input image size
    output = np.zeros((ih, iw), dtype="float32")
    # y is sliding window downwards and x is sliding window leftwards
    for y in np.arange(pad, ih + pad):
        for x in np.arange(pad, iw + pad):
            # getting region of interest also called neighbourhood (here window) over which
            # convolution is to be applied
            # getting coordinates from extreme left to extreme right as row (y),
            # and extreme up to extreme bottom of window as column (x)
            window = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # applying product element wise and taking overall sum of all products
            result = (window * filter).sum()
            # putting the calculated value at the appropriate position of output matrix
            output[y - pad, x - pad] = result
    # rescaling the final output values
    output = rescale_intensity(output, in_range=(0, 255))
    # rounding off the values to get actual intensity
    output = (output * 255).astype("uint8")
    return output

def output_image(gray, kernelBank, choice='convolution'):
    '''
    Parameters:
    gray : input image matrix (numpy array) , now gray scaled
    kernelBank : tuples including first item as name we provide to filter and other item is actual matrix of filter values
    choice : indicates what function to use either convolution or opencv (by default : convolution is chosen)

    '''
    # plotting all images on same figure based on choice provided
    if choice == 'convolution' or choice == 'opencv':
        fig=plt.figure(figsize=(30, 30))
        i = 1
        for (kernelName, kernel) in kernelBank:
            print(f"Processing {kernelName} operation....")
            if choice ==  'convolution':
                # apply custom convolution
                Output = convolution(gray, kernel)
            else:
                # apply opencv convolution
                Output = cv2.filter2D(gray, -1, kernel)
            fig.add_subplot(2, 3, i)
            i += 1
            plt.imshow(Output, cmap='gray')
            if choice ==  'convolution':
                plt.title(f"{kernelName} : using convolution")
            else:
                plt.title(f"{kernelName} : using opencv")

        print("Outputting the final images...")
        plt.show()
    else:
        print("Please enter correct choice : <convolution> or <opencv>")

def convolution_main(image_path, choice='convolution'):
    '''
    Parameters:
    image_path : input image full path
    choice : indicates what function to use either convolution or opencv (by default : convolution is chosen)

    '''
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    sharpen = np.array((
    	[0, -1, 0],
    	[-1, 5, -1],
    	[0, -1, 0]), dtype="int")
    laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
    sobelX = np.array((
    	[-1, 0, 1],
    	[-2, 0, 2],
    	[-1, 0, 1]), dtype="int")
    sobelY = np.array((
    	[-1, -2, -1],
    	[0, 0, 0],
    	[1, 2, 1]), dtype="int")
    kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY)
    )
    # reading the image
    image = cv2.imread(image_path)
    # changing image to gray scale , that means now the channel is 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image read , shape of image is {image.shape}")
    # calling for applying filters and plotting the output images
    output_image(gray, kernelBank, choice)

if __name__ == '__main__':
    print("WELCOME TO IMAGE PROCESSING FUNCTIONS LAB !")
    print("Currently available options for processing are either using custom convolution function or using opencv filter function !")
    print("PUT THE ARGUMENTS IN ORDER : python convolution.py <outputPath> <convolution/opencv>")
    if len(sys.argv) > 1:
        if len(sys.argv) == 3:
            convolution_main(sys.argv[1], sys.argv[2])
        else:
            convolution_main(sys.argv[1], 'convolution')
    else:
        print("Please input arguments - path of image correctly !")
