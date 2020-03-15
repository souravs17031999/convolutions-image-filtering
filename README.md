# convolutions-image-filtering
This Repo showcases command line script to output images after applying filters using convolution operation.

# Getting Started Locally : 
> Using cmd : 
```
python convolution.py <outputPath> <convolution/opencv>
```    
* Note : convolution uses basic custom built function for applying convolution operation using sliding window approach whereas opencv uses convolution using it's optimized filter2D function.    
ex.   
```
python convolution.py C:/Users/DELL/desktop/2.JPG convolution
```   
![conv image](/images/convolution.JPG)    
```
python convolution.py C:/Users/DELL/desktop/2.JPG opencv
``` 
![opencv image](/images/opencv.JPG)    

