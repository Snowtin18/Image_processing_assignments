import numpy as np
import cv2
from sympy import symbols, Eq, solve 




#Setting value of a
a=2



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
    s=np.sum(h)
    h=h/s
    return h

#Caluculation of sigma value, which is a gaussian distribution
def sigma_calculation(m,n,N):
    b=(N**2)/10.59 #Based on the boundary conditions the value of a and b is found
    sigma=a*((np.e)**((-(((m-(N/2))**2)+((n-(N/2))**2)))/b))

    return sigma

#Generating kernal using gaussian values
def kernel_generation(m,n,N):
    sigma=sigma_calculation(m,n,N)
    return gaussian(sigma)

#Sliding kernal method
def conv(img):

    # Perform convolution
    output = np.zeros_like(img)
    for i in range(output.shape[0]): #Here i,j are m,n
        for j in range(output.shape[1]):
            # Generating the kernal
            kernel=kernel_generation(i,j,output.shape[0])
            # Get the dimensions of the kernel
            kernel_rows, kernel_cols = kernel.shape
            # Calculate padding size
            pad_rows = kernel_rows // 2
            pad_cols = kernel_cols // 2
            # Pad the image
            img_padded = np.pad(img, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')
            # Flip the kernel horizontally and vertically
            kernel = np.flipud(np.fliplr(kernel))
            output[i, j] = np.sum(kernel * img_padded[i:i+kernel_rows, j:j+kernel_cols])
    return output


# Read the input image
img_mat1 = cv2.imread('Globe.png', cv2.IMREAD_GRAYSCALE)

#Perform covolution

output=conv(img_mat1)


cv2.imwrite('output.png', output)

