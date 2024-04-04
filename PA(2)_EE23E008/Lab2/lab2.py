#Required packages
import matplotlib.image as img
import numpy as np
from sympy import symbols, Eq, solve 

#Symbols
a, b, tx, ty = symbols('a,b,tx,ty')

#Read the image to convert it into matrix form
img_mat=np.array(img.imread('IMG2.png'))

#Finding R and T
eq1 = Eq((93*a+248*b+tx), 29)
eq2 = Eq((248*a-93*b+ty), 124)
eq3 = Eq((328*a+399*b+tx), 157)
eq4 = Eq((399*a-328*b+ty), 372)

print("Equation 1:")
print(eq1)
print("Equation 2:")
print(eq2)
print("Equation 3:")
print(eq3)
print("Equation 4:")
print(eq4)

solutions=solve((eq1, eq2,eq3,eq4), (a, b,tx,ty))

#The solutions
asol=float(solutions[a])
bsol=float(solutions[b])
txsol=float(solutions[tx])
tysol=float(solutions[ty])


#Creating empty matrix
target=np.zeros((517,598))

#Performing rotation and translation using the solutions found above
for i in range(0,517):
    for j in range(0,598):
        #Perform rotation and translation instead after finding R and T
        #We are here performing a target to source transform
        xs=(i)*asol-(j)*bsol-txsol
        ys=(i)*(bsol)+(j)*asol-tysol
        xs_=int(np.floor(xs))
        ys_=int(np.floor(ys))

        d=xs-xs_
        c=ys-ys_
        #print(xs_,ys_)
        
        if(xs_<516 and xs_>0 and ys_<597 and ys_>0):
            #Bilinear Interpolation
            target[i][j]=(1-c)*(1-d)*img_mat[xs_][ys_]+(1-c)*d*img_mat[xs_][ys_+1]+c*(1-d)*img_mat[xs_+1][ys_]+c*d*img_mat[xs_+1][ys_+1]

        else:
            target[i][j]=0

#The rotated image is saved as rotated_img2.png
img.imsave('rotated_img2.png',target)

#Coverting image1 to matrix form
source_read=np.array(img.imread('IMG1.png'))

source=np.zeros((517,598))

#Copying the image matrix to the source matrix
for i in range(0,296):#x-coordinates
    for j in range(0,512):#y-coordinates
        xt,yt=i,j #Column,row-wise
        source[int(xt+110)][int(yt+43)]=source_read[i][j]

#This is the source image
img.imsave('source.png',source)

#Aligning source and target
#The below code is to find the first pixel of source and the rotated image.
row,col=0,0
found=False
for i in target:
    for j in i:
        if(j>=0.1):
            print('Pixel value:',j)
            print("Target 1st pixel:")
            print(row,col)
            found=True
            break
        row+=1
    if(found):
            break
    row=0
    col+=1

row,col=0,0
found=False
for i in source:
    for j in i:
        if(j!=0):
            print("Source 1st pixel:")
            print(row,col)
            found=True
            break
        row+=1
    if(found):
            break
    row=0
    col+=1

#Need to move this image by row,col
#By using the difference between the row, col values found above. We shift the rotated image to align it with source image
aligned_target=np.zeros((517,598))
for i in range(0,517):
    for j in range(0,598):
        try:
            aligned_target[i][j]=target[i-55][j+15]
        except:
            aligned_target[i][j]=0

#This is the aligned image
img.imsave('aligned_target.png',aligned_target)

#We subtract the source image and the rotated-aligned image to find the difference
change=source-aligned_target

#This is the final solution. The change in the image
img.imsave('changed.png',change)

