# Import necessary libraries
from sift import sift as sift_corresp
import numpy as np
from sympy import symbols, Eq, solve,Matrix
import cv2

from random import sample


eps1=0.001
eps2=25

# Function to form part of the matrix A for homography estimation
def form_part_a_mat(x2,y2,x1,y1):#Output,Input
    h=[[-x1,-y1,-1,0,0,0,x1*x2,x2*y1,x2],
       [0,0,0,-x1,-y1,-1,y2*x1,y2*y1,y2]]
    
    return h

# Function to form the matrix A for homography estimation using point correspondences
#Do row stack of h1,h2,h3,h4
def form_a_mat(corres):

    #For h1
    x2,y2=corres[0][0]
    x1,y1=corres[0][1]
    h1=form_part_a_mat(x2,y2,x1,y1)

    #For h1
    x2,y2=corres[1][0]
    x1,y1=corres[1][1]
    h2=form_part_a_mat(x2,y2,x1,y1)

    #For h1
    x2,y2=corres[2][0]
    x1,y1=corres[2][1]
    h3=form_part_a_mat(x2,y2,x1,y1)

    #For h1
    x2,y2=corres[3][0]
    x1,y1=corres[3][1]
    h4=form_part_a_mat(x2,y2,x1,y1)

    return np.ma.row_stack([h1,h2,h3,h4])

# Function to compute the null space of a matrix using SVD
def nullspace(a):

    u,s,v=np.linalg.svd(a)
    return np.squeeze(v[:][-1])

# Function to form the homography matrix from the solution
def form_homo_mat(sol):

    homomat=np.array([[sol[0],sol[1],sol[2]],
                      [sol[3],sol[4],sol[5]],
                      [sol[6],sol[7],sol[8]]])
    
    return homomat

# Function to estimate homography using RANSAC algorithm
def ransac(corresp1,corresp2):

    #Random sampling
    corres=list(zip(corresp1,corresp2))
    iterations=len(corres)
    max_consensus=[]
    max_H=np.array([[]])
    eps2=len(corresp1)*0.8# Set threshold for minimum consensus
    eps1=10# Set threshold for point distance

    while(iterations>0): # Loop over RANSAC iterations
        rand_sample=sample(corres,4) # Randomly sample 4 point correspondences

        #print(rand_sample)

        h=form_a_mat(rand_sample) # Form the matrix A for homography estimation

        sol=nullspace(h) # Compute the null space of matrix A to get the solution vector

        H=form_homo_mat(sol) # Form the homography matrix from the solution vector
        consensus=[] # Initialize variable to store consensus points
        for i in corres: # Loop over all point correspondences
            
            target_coord=np.array(i[0]) # Extract target coordinates
            source_coord=np.array(i[1]) # Extract source coordinates
            source_coord=np.append(source_coord,1) # Append 1 for homogeneous coordinate
            target_coord=np.append(target_coord,1) # Append 1 for homogeneous coordinate

            pred_coord=np.matmul(H,source_coord) # Predict target coordinates using homography

            pred_coord[0]=pred_coord[0]/pred_coord[2]
            pred_coord[1]=pred_coord[1]/pred_coord[2]
            pred_coord[2]=1
            # print('pred coord:',pred_coord)
            dist=np.linalg.norm(pred_coord-target_coord) # Compute distance between predicted and actual coordinates
            # print('dist:',dist)
            if(dist<eps1): # Check if distance is below threshold
                
                consensus.append(i) # Add point correspondence to consensus

            if(len(consensus)>eps2): # Check if consensus is above threshold
                print("Found a good enough H") # Return the homography if consensus is sufficient
                return H
            else:
                if(len(max_consensus)<len(consensus)): # Update maximum consensus if necessary
                    max_consensus=consensus
                    max_H=H
        iterations-=1
    print('Returning maximum H')
    return max_H



#     #A function that gives homography when give 4 pt correspondance
# Read the images
# img_mat1=np.array(cv2.imread('img1.png'))
# img_mat2=np.array(cv2.imread('img2.png'))
# img_mat3=np.array(cv2.imread('img3.png'))

img_mat1=np.array(cv2.imread('edimg1.png'))
img_mat2=np.array(cv2.imread('edimg2.png'))
img_mat3=np.array(cv2.imread('edimg3.png'))

# Perform feature matching using SIFT
[corresp1_img2,corresp1_img1] = sift_corresp(img_mat2,img_mat1)
h21=ransac(corresp1_img2,corresp1_img1) # Estimate homography between img2 and img1

# Match features between img3 and img1
[corresp2_img2,corresp2_img1] = sift_corresp(img_mat3,img_mat1)

h31=ransac(corresp2_img2,corresp2_img1)  # Estimate homography between img3 and img1

# Create a canvas to stitch images
canvas=np.zeros((360,2000,3))

for i in range(0,360):#x-coordinates
    for j in range(0,2000):#y-coordinates
        xt,yt=i,j#Column,row-wise
        homo_coord=np.array([[xt],
                             [yt],
                             [1]])
        #Value from image 1
        try:
            img1_val=img_mat1[xt][yt][0]
        except:
            img1_val=0

        #Value from image 2
        #Transormation to 1st image frame
        homo_coord21=np.matmul(h21,homo_coord)

        homo_coord21[0]=(homo_coord21[0]/homo_coord21[2])
        homo_coord21[1]=(homo_coord21[1]/homo_coord21[2])
        homo_coord21[2]=1
        
        try:
            xs2=homo_coord21[0]
            ys2=homo_coord21[1]
            xs_2=int(np.floor(xs2))
            ys_2=int(np.floor(ys2))

            d=xs2-xs_2
            c=ys2-ys_2
            
            if(xs_2<359 and xs_2>0 and ys_2<639 and ys_2>0):
                #Bilinear Interpolation
                img2_val=(1-c)*(1-d)*img_mat2[xs_2][ys_2][0]+(1-c)*d*img_mat2[xs_2][ys_2+1][0]+c*(1-d)*img_mat2[xs_2+1][ys_2][0]+c*d*img_mat2[xs_2+1][ys_2+1][0]

            else:
                img2_val=0
        except Exception as err:
            img2_val=0

        #Value from image 3
        #Transormation to 1st image frame
        homo_coord31=np.matmul(h31,homo_coord)

        homo_coord31[0]=int(homo_coord31[0]/homo_coord31[2])
        homo_coord31[1]=int(homo_coord31[1]/homo_coord31[2])
        homo_coord31[2]=1

        try:
            xs3=homo_coord31[0]
            ys3=homo_coord31[1]
            xs_3=int(np.floor(xs3))
            ys_3=int(np.floor(ys3))

            d=xs3-xs_3
            c=ys3-ys_3
            
            if(xs_3<359 and xs_3>0 and ys_3<639 and ys_3>0):
                #Bilinear Interpolation
                img3_val=(1-c)*(1-d)*img_mat3[xs_3][ys_3][0]+(1-c)*d*img_mat3[xs_3][ys_3+1][0]+c*(1-d)*img_mat3[xs_3+1][ys_3][0]+c*d*img_mat3[xs_3+1][ys_3+1][0]

            else:
                img3_val=0
        except:
            img3_val=0

        #Assigning canvas values as avg of the values from 3 images
        canvas[int(xt)][int(yt)]=int(((img1_val+img2_val+img3_val))/3)
        # canvas[int(xt)][int(yt)]=((img1_val+img2_val))/2

# cv2.imwrite('edCanvas.png',canvas)
cv2.imwrite('edCanvas.png',canvas)

