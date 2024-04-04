#Required packages
import matplotlib.image as img
import numpy as np

#Read the image to convert it into matrix form
img_mat=np.array(img.imread("pisa_rotate.png"))

print(np.shape(img_mat))
#Creating an empty matrix for source of size 512x512 for extra space.
source=np.zeros((1024,1024))
#Copying the image matrix to the source matrix
for i in range(0,482):
    for j in range(0,207):
        xt,yt=i+271,j+409
        source[int(xt)][int(yt)]=img_mat[i][j]
#Saving the source image
img.imsave('q2_source.png',source)

#Creating an empty matrix for target of size 512x512 for extra space.
target=np.zeros((1024,1024))
#Rotation angle
theta=(np.pi/180)*-2

for i in range(271,271+482): #Traversing along x
    for j in range(409,409+207): #Traversing along y
        #ROTATION
        xt=(i)*np.cos(theta)+(j)*np.sin(theta)
        yt=(i)*(-np.sin(theta))+(j)*np.cos(theta)
        #Converting decimals to integers
        xt_=np.floor(xt)
        yt_=np.floor(yt)
        print(xt_,yt_)
        print(source[i][j])
        #Filling the target matrix
        target[int(xt_)][int(yt_)]=source[i][j]

img.imsave('q2_target.png',target) #Saving the target image



