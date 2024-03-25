# %%
import pyrealsense2 as rs
import numpy as np
import cv2

# %%
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# %%
while True:


    # %% Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # %% Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # print(depth_image.max())

    # %% Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # print(depth_colormap.shape)
    depth_image = (depth_image/65535*255).astype('uint8')
    depth_image = depth_image.reshape(480, 640, 1)
    depth_image = np.repeat(depth_image,3, axis=2)
    depth_colormap = depth_image
    # depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    print(color_image.shape, depth_colormap.shape)

    images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

pipeline.stop()