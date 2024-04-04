import numpy as np
import cv2


# Define Gaussian function to compute the value of the Gaussian distribution
def gaussian_value(x,y,midx,midy,sigma):
    h=(1/(2*np.pi*(sigma**2)))*(np.e**(-((midx-x)**2 + (midy-y)**2)/(2*(sigma**2))))
    return h

# Generate Gaussian kernel
def gaussian(sigma):
    # Compute the size of the kernel based on sigma
    h=np.zeros([int(np.ceil(6*sigma+1)),int(np.ceil(6*sigma+1))])
    shape=np.shape(h)
    midx=((shape[0]+1)//2)-1
    midy=((shape[1]+1)//2)-1
    #traverse through h and appply gaussian_value function to find value
    for i in range(shape[0]):
        for j in range(shape[1]):
            h[i,j]=gaussian_value(i,j,midx,midy,sigma)
    return h

# def conv(a,b):

#     #My attempt

#     rowa,cola=np.shape(a)
#     rowb,colb=np.shape(b)

#     rowo=rowa+rowb-1
#     colo=cola+colb-1
#     output=np.zeros((rowo,colo))
#     print(np.shape(output))

#     a=np.vstack((a,np.zeros((rowo-rowa,cola))))
#     rowa+=rowo-rowa
#     a=np.hstack((a,np.zeros((rowa,colo-cola))))
#     cola+=colo-cola

#     b=np.vstack((b,np.zeros((rowo-rowb,colb))))
#     rowb+=rowo-rowb
#     b=np.hstack((b,np.zeros((rowb,colo-colb))))
#     rowb+=colo-colb


#     for m in range(0,rowo):
#         for n in range(0,colo):
            

#             #The summation
#             for m0 in range(0,rowo):
#                 for n0 in range(0,colo):
#                     output[m,n]+=a[m0,n0]*b[m-m0,n-n0]

#     return output

#Sliding kernal method
def conv(img, kernel):
    # Get the dimensions of the kernel
    kernel_rows, kernel_cols = kernel.shape
    # Calculate padding size
    pad_rows = kernel_rows // 2
    pad_cols = kernel_cols // 2
    # Pad the image
    img_padded = np.pad(img, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')
    # Flip the kernel horizontally and vertically
    kernel = np.flipud(np.fliplr(kernel))
    # Perform convolution
    output = np.zeros_like(img)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(kernel * img_padded[i:i+kernel_rows, j:j+kernel_cols])
    return output


# Read the input image
img_mat1 = cv2.imread('Mandrill.png', cv2.IMREAD_GRAYSCALE)

# Generate Gaussian kernel
sigma = 0.3
h = gaussian(sigma)

# Perform convolution
output = conv(img_mat1, h)

# Save the output image
cv2.imwrite('Output.png', output)
