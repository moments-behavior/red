import os
import argparse
import glob
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,
                    help="Directory of .mp4 files OR a single .mp4 file.")
parser.add_argument('-o', '--output_dir', type=str, default="test")
parser.add_argument('-s', '--frame_start', type=int, default=0,
                    help="Start frame (inclusive). Default 0.")
parser.add_argument('-e', '--frame_end', type=int, default=-1,
                    help="End frame (inclusive). -1 means the last frame "
                    "of each video. Default -1.")

# list all cameras, and then load videos, and then save out
args = parser.parse_args()
input_path = args.input
output_dir = args.output_dir
frame_start = args.frame_start
frame_end = args.frame_end

if os.path.isfile(input_path):
    root_dir = os.path.dirname(input_path) or "."
    cam_names = [os.path.splitext(os.path.basename(input_path))[0]]
else:
    root_dir = input_path
    cam_names = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(root_dir, "*.mp4"))
    )

save_dir = os.path.join(root_dir, output_dir)
os.makedirs(save_dir, exist_ok=True)

for i, cam in enumerate(cam_names):
    print(i)
    output_name = os.path.join(save_dir, "{}.mp4".format(cam))
    video_file = os.path.join(root_dir, "{}.mp4".format(cam))
    video = cv.VideoCapture(video_file)
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    end = total - 1 if frame_end < 0 else frame_end
    video.set(cv.CAP_PROP_POS_FRAMES, frame_start)
    width  = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = video.get(cv.CAP_PROP_FPS)
    output_dims = (width, height)
    writer = cv.VideoWriter(output_name, cv.VideoWriter_fourcc('a','v','c','1'), fps, output_dims)

    for frame_num in range(frame_start, end + 1):
        ret, frame = video.read()
        writer.write(frame)

    writer.release() 
    video.release() 