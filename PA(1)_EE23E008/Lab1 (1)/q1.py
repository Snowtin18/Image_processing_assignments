#Required packages
import matplotlib.image as img
import numpy as np

#Read the image to convert it into matrix form
img_mat=np.array(img.imread("lena_translate.png"))

#Creating an empty matrix for source of size 512x512 for extra space.
source=np.zeros((512,512))
#Copying the image matrix to the source matrix
for i in range(0,256):
    for j in range(0,256):
        xt,yt=i,j
        source[int(xt)][int(yt)]=img_mat[i][j]
#Saving the source image
img.imsave('q1_source.png',source)

#Creating an empty matrix for target of size 512x512 for extra space.
target=np.zeros((512,512))


for i in range(0,256): #Traversing along x
    for j in range(0,256): #Traversing along y
        xt,yt=i+3.75,j+4.3 #TRANSLATION
        xt_=np.floor(xt) #Converting decimals to integers
        yt_=np.floor(yt)
        print(xt_,yt_)
        print(source[i][j]) 
        target[int(xt_)][int(yt_)]=source[i][j] #Filling the target matrix

img.imsave('q1_target.png',target) #Saving the target image



