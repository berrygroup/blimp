from skimage import io
from pystackreg import StackReg

# conda install pystackreg -c conda-forge

img0 = io.imread("some_multiframe_image.tif")  # 3 dimensions : frames x width x height

sr = StackReg(StackReg.RIGID_BODY)

# register each frame to the previous (already registered) one
# this is what the original StackReg ImageJ plugin uses
out_previous = sr.register_transform_stack(img0, reference="previous")

# register to first image
out_first = sr.register_transform_stack(img0, reference="first")

# register to mean image
out_mean = sr.register_transform_stack(img0, reference="mean")

# register to mean of first 10 images
out_first10 = sr.register_transform_stack(img0, reference="first", n_frames=10)

# calculate a moving average of 10 images, then register the moving average to the mean of
# the first 10 images and transform the original image (not the moving average)
out_moving10 = sr.register_transform_stack(img0, reference="first", n_frames=10, moving_average=10)
