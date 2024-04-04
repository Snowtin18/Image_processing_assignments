#Required packages
import matplotlib.image as img
import numpy as np

#Read the image to convert it into matrix form
img_mat=np.array(img.imread('cells_scale.png'))

print(np.shape(img_mat))
#Creating an empty matrix for source of size 512x512 for extra space.
source=np.zeros((1024,1024))
#Copying the image matrix to the source matrix
for i in range(0,240):#x-coordinates
    for j in range(0,315):#y-coordinates
        xt,yt=i,j #Column,row-wise
        source[int(xt)][int(yt)]=img_mat[i][j]
#Saving the source image
img.imsave('q3_source.png',source)

#Creating an empty matrix for target of size 512x512 for extra space.
target=np.zeros((512,512))

for i in range(0,512):
    for j in range(0,512):
        #SCALING
        #We are here performing a target to source transform
        xs,ys=(i/0.8),(j/1.3)
        xs_=int(np.floor(xs))
        ys_=int(np.floor(ys))

        b=xs-xs_
        a=ys-ys_
        print(xs_,ys_)
        
        if(xs_<239 and ys_<314):
            #Bilinear Interpolation
            target[i][j]=(1-a)*(1-b)*img_mat[xs_][ys_]+(1-a)*b*img_mat[xs_][ys_+1]+a*(1-b)*img_mat[xs_+1][ys_]+a*b*img_mat[xs_+1][ys_+1]

        else:
            target[i][j]=0

img.imsave('q3_target.png',target)



