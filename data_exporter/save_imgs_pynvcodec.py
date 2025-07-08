import numpy as np
import cv2
import ctypes as C


def cast_address_to_1d_bytearray(base_address, size):
    return np.ctypeslib.as_array(
        C.cast(base_address, C.POINTER(C.c_uint8)), shape=(size,)
    )


input_file_path = (
    "/nfs/exports/ratlv/exp_2025_03/unsorted/2025_07_05_23_33_14/Cam710038.mp4"
)

# input_file_path = (
#     "/nfs/exports/ratlv/exp_2025_03/unsorted/2025_07_05_17_38_30/Cam710038.mp4"
# )
# input_file_path = "/nfs/exports/ratlv/exp_2025_03/remy_emilie/2025_06_26_16_25_51/Cam710038.mp4"
# opencv
cap = cv2.VideoCapture(input_file_path)  # Or use 'video.mp4' for a file

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the FPS
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

frame_number = 181
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
if ret:
    print("Successful seek.")
else:
    print("Failed to retrieve the frame.")
# Release the stream
cap.release()


import PyNvVideoCodec as nvc

# Input video
use_device_memory = 0  # 0 = system memory, 1 = device memory

# Initialize decoder
simple_decoder = nvc.SimpleDecoder(
    input_file_path,
    gpu_id=0,
    use_device_memory=use_device_memory,
    output_color_type=nvc.OutputColorType.RGB,
)

# Get video metadata
metadata = simple_decoder.get_stream_metadata()
print(metadata)
print(f"Total frames: {metadata.num_frames}")
print(f"FPS: {metadata.average_fps}")

# Frame indices to retrieve
frame_indices = [0, 5]

# Decode specific frames
frames = simple_decoder.get_batch_frames_by_index(frame_indices)
luma_base_addr = frames[0].GetPtrToPlane(0)
new_array = cast_address_to_1d_bytearray(
    base_address=luma_base_addr, size=frames[0].framesize()
)


img = new_array.reshape(
    (metadata.height, metadata.width, -3)
)  # or (height, width) for grayscale
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite("test_imgs.png", img_bgr)

# import pdb

# pdb.set_trace()
# # Save each frame to disk
# for idx, frame in zip(frame_indices, frames):
#     # Convert NV12 to BGR for saving
#     bgr_frame = frame.get_bgr()  # returns numpy.ndarray in HWC format
#     filename = f"frame_{idx:04d}.png"
#     print(f"Saved {filename}")
