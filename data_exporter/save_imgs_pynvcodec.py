import PyNvVideoCodec as nvc

input_file_path = "/mnt/data/swc_snippet/Cam2010126.mp4"
use_device_memory = 0
simple_decoder = nvc.SimpleDecoder(
    input_file_path, gpu_id=0, use_device_memory=use_device_memory
)

# Get video metadata
metadata = simple_decoder.get_stream_metadata()
total_frames = metadata.num_frames
video_fps = metadata.average_fps
print(total_frames)
print(video_fps)
