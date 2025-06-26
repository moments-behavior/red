import argparse
import glob
import os
import cv2 as cv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_folder', type=str, required=True)
parser.add_argument('-fs', '--frame_start', type=int, required=True)
parser.add_argument('-fe', '--frame_end', type=int, required=True)
parser.add_argument('-o', '--output_folder', type=str, required=True)

args = parser.parse_args()
input_folder = args.input_folder
frame_start = args.frame_start
frame_end = args.frame_end
number_of_frames = frame_end - frame_start + 1
output_folder = args.output_folder

cam_names = []
for file in glob.glob(input_folder + "/*.mp4"):
    file_name = file.split("/")
    cam_names.append(file_name[-1].split(".")[0])
cam_names.sort()

for cam_name in cam_names:
    video_file = input_folder + "/{}.mp4".format(cam_name)
    video = cv.VideoCapture(video_file)
    video.set(cv.CAP_PROP_POS_FRAMES, frame_start)
    frame_rate = video.get(cv.CAP_PROP_FPS)

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter(
        "{}/{}.mp4".format(output_folder, cam_name), fourcc, frame_rate, (2200, 3208))

    for i in tqdm(range(number_of_frames)):
        ret, frame = video.read()
        output.write(frame)

    output.release()
    video.release()
