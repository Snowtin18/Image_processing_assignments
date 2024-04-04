import scipy.io
import numpy as np
from itertools import islice
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




def conv(image,kernal):

    # Perform convolution
    output = np.zeros_like(image)
    for i in range(output.shape[0]): #Here i,j are m,n
        for j in range(output.shape[1]):
            # Generating the kernal
            kernel=kernal
            # Get the dimensions of the kernel
            kernel_rows, kernel_cols = kernel.shape
            # Calculate padding size
            pad_rows = kernel_rows // 2
            pad_cols = kernel_cols // 2
            # Pad the image
            image_padded = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')
            # Flip the kernel horizontally and vertically
            kernel = np.flipud(np.fliplr(kernel))
            output[i, j] = np.sum(kernel * image_padded[i:i+kernel_rows, j:j+kernel_cols])
    return output

def focus_operator(image):
    #Define the kernal
    kernal_x=np.array([[0,0,0],
                     [1,-2,1],
                     [0,0,0]])
    
    kernal_y=np.array([[0,1,0],
                     [0,-2,0],
                     [0,1,0]])
    
    kernal_sum=np.ones((q,q))
    
    #Covolve image with kernal
    image_xx=conv(image,kernal_x)

    image_yy=conv(image,kernal_y)

    image_xx_mod=np.abs(image_xx)
    image_yy_mod=np.abs(image_yy)

    image_ml=image_xx_mod+image_yy_mod

    image_sml=conv(image_ml,kernal_sum)

    return image_sml

mat = scipy.io.loadmat('stack.mat')

cv2.imwrite('image1.png',mat['frame001'])

#Global variables
deld=50.50
q=1
#Dictionary processing

for i in range(0,4):
    del mat[next(islice(mat, 0, None))]

mat.pop('s')

#Applying focus operator on the stack of images

mat_operated=dict()

for key, image in mat.items() :
    print(key)
    mat_operated[key]=focus_operator(image)

#Creating a depth image

mat_shape=np.shape(mat_operated['frame099'])

image_depth=np.zeros((mat_shape[0],mat_shape[1]))

frame_count=5



for i in range(mat_shape[0]):
    for j in range(mat_shape[1]):
        f_values=[mat_operated[f'frame{95+k+1:03d}'][i][j] for k in range(frame_count)]
        print(f_values)
        
        #m
        fm=max(f_values)
        fm_index=f_values.index(fm)
        dm=fm_index*deld
        print(fm_index)
        #m-1
        fm_prev_index=fm_index-1
        dm_prev=fm_prev_index*deld
        fm_prev=f_values[fm_prev_index]
        #m+1
        fm_after_index=fm_index+1
        if(fm_after_index>=len(f_values)):
            fm_after_index=0
            dm_after=0
            fm_after=0
        else:
            dm_after=fm_after_index*deld
            fm_after=f_values[fm_after_index]

        #depth image
        dbar=(((np.log(fm)-np.log(fm_prev))*((dm_after**2)-(dm**2))-(np.log(fm)-np.log(fm_after))*((dm_prev**2)-(dm**2)))/(2*(deld)*((2*(np.log(fm))-(np.log(fm_after))-(np.log(fm_prev))))))
        if(dbar==np.nan):
            dbar=0
        image_depth[i][j]=dbar

cv2.imwrite('image_depth.png',image_depth)

# Create meshgrid for 3D plot
x = np.arange(image_depth.shape[1])
y = np.arange(image_depth.shape[0])
X, Y = np.meshgrid(x, y)

# Flatten the arrays to create a scatter plot
x_flat = X.flatten()
y_flat = Y.flatten()
depth_flat = image_depth.flatten()

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_flat, y_flat, depth_flat, c=depth_flat, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')
ax.set_title('3D Depth Scatter Plot')

plt.savefig('detph_map.png')
plt.show()





