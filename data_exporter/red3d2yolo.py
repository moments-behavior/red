import cv2 as cv
import os
from utils import *
from keypoints import *
import argparse
from datetime import datetime
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--label_folder', type=str, required=True)
parser.add_argument('-m', '--mode', type=str, default='point2bbox')
parser.add_argument('-o', '--output_folder', type=str, required=True)


args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
det_mode = args.mode
output_folder = args.output_folder


cameras = get_all_cams_in_labeled_folder(label_folder)


# Save annotations
world_point_folder = label_folder + "/worldKeyPoints"
all_files = glob.glob(world_point_folder + "/*")
all_files.sort()
select_most_recent_labels = all_files[-1]
print("Select most recent label: {}".format(select_most_recent_labels))

selected_annotation = select_most_recent_labels.split("/")[-1][10:]
selected_annotation = selected_annotation.split(".")[0]
world_labels = csv_reader_rats(select_most_recent_labels, 1, three_d=True) 

# filter out invalid lables
world_labels_filterd = {}
for name, value in world_labels.items():
    if not np.any(value==1E7):
        world_labels_filterd[name] = value

labels_frames = np.asarray(list(world_labels_filterd.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
np.random.shuffle(id_shuffled)
num_train = int(np.floor(total_num_labels * 0.9))
print("Train set: {}, validation set: {}.".format(num_train, total_num_labels - num_train))
train_ids = id_shuffled[:num_train]
train_ids = np.sort(train_ids)
val_ids = id_shuffled[num_train:]
val_ids = np.sort(val_ids)
## split frames to train and val
train_image_frames = labels_frames[train_ids]
val_image_frames = labels_frames[val_ids]


trial_name = selected_annotation

num_keypoints = 1   
id_ball = 1
d_ball = 100
## 
annotations = process_one_session_ball(trial_name, label_folder, num_keypoints, selected_annotation, train_image_frames, cameras,d_ball,id_ball)
create_yolo_annotation_files(output_folder,trial_name,annotations,cameras,"train")

annotations = process_one_session_ball(trial_name, label_folder, num_keypoints, selected_annotation, val_image_frames, cameras,d_ball,id_ball)
create_yolo_annotation_files(output_folder,trial_name,annotations,cameras,"valid")


exit()
# use the most recent file for labels 
if det_mode == 'point2bbox':
    labels_file = sorted(glob.glob(root_dir + "/ball_labeled_data/{}/*".format(cam_name)))[-1]

print("Loading most recent file: ", labels_file)
video_file = root_dir + "/movies/{}.mp4".format(cam_name)

labels_unfiltered = csv_reader_rodent(labels_file, num_keypoints, False, [0, 1, 2, 3, 4, 5])

labels = {}
for key, value in labels_unfiltered.items():
    if not((value >= 1E7).any()):
        labels[key] = value    

labels_frames = np.asarray(list(labels.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
#np.random.shuffle(id_shuffled)
num_train = int(np.floor(total_num_labels * 0.9))
print("Train set: {}, validation set: {}.".format(num_train, total_num_labels - num_train))

train_ids = id_shuffled[:num_train]
np.sort(train_ids)
val_ids = id_shuffled[num_train:]
np.sort(val_ids)

video = cv.VideoCapture(video_file)
amount_of_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
fps = video.get(cv.CAP_PROP_FPS);
print("Number of frames: {}, framerate {}".format(amount_of_frames, fps))
frame_width  = video.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = video.get(cv.CAP_PROP_FRAME_HEIGHT)


vr = VideoReader(video_file, ctx=cpu(0))

image_save_dir_train = output_dir + "images/train/"
label_save_dir_train = output_dir + "labels/train/"

image_save_dir_val = output_dir + "images/val/"
label_save_dir_val = output_dir + "labels/val/"

os.makedirs(image_save_dir_train, exist_ok=True)
os.makedirs(label_save_dir_train, exist_ok=True)
os.makedirs(image_save_dir_val, exist_ok=True)
os.makedirs(label_save_dir_val, exist_ok=True)

save_format = "jpg"

def create_yolo_ball_bb_line(ball_keypoints, frame_width, frame_height, ball_bb_size, label_idx):
    ball_center_x = ball_keypoints[0] / frame_width
    ball_center_y = ball_keypoints[1] / frame_height
    ball_w = ball_bb_size / frame_width
    ball_h = ball_bb_size / frame_height
    line = "{} {} {} {} {}".format(label_idx, ball_center_x, ball_center_y, ball_w, ball_h)
    return line


def create_yolo_rat_bb_line(rat_keypoints, frame_width, frame_height, margin, label_idx):
    rat_x = rat_keypoints[:, 0] / frame_width
    rat_y = rat_keypoints[:, 1] / frame_height
    
    margin_x = margin / frame_width
    margin_y = margin / frame_height
    rat_x_min = np.max((0, np.min(rat_x) - margin_x))
    rat_x_max = np.min((frame_width, np.max(rat_x) + margin_x))
    rat_y_min = np.max((0, np.min(rat_y) - margin_y))
    rat_y_max = np.min((frame_height, np.max(rat_y) + margin_y))
    rat_center_x = (rat_x_min + rat_x_max) / 2.0 
    rat_center_y = (rat_y_min + rat_y_max) / 2.0
    rat_w = rat_x_max - rat_x_min
    rat_h = rat_y_max - rat_y_min

    line = "{} {} {} {} {}".format(label_idx, rat_center_x,rat_center_y,rat_w,rat_h)
    return line

print("Create train dataset...")
# TODO: could load in smaller batches for less memory 
train_frame_ids = labels_frames[train_ids]
batch_frames = vr.get_batch(train_frame_ids)
batch_frames = batch_frames.asnumpy()
for idx in range(train_ids.shape[0]):
    frame_idx = train_frame_ids[idx]
    frame = batch_frames[idx]
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imwrite(image_save_dir_train + "{}.{}".format(frame_idx, save_format), frame)
    with open(os.path.join(label_save_dir_train, str(frame_idx) + '.txt'), 'w') as f:
        line = create_yolo_rat_bb_line(labels[frame_idx], frame_width, frame_height, 40, 0)        
        f.write(line)
        f.write('\n')


print("Create val dataset...")
val_frame_ids = labels_frames[val_ids]
batch_frames = vr.get_batch(val_frame_ids)
batch_frames = batch_frames.asnumpy()
for idx in range(val_ids.shape[0]):
    frame_idx = val_frame_ids[idx]
    frame = batch_frames[idx]
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imwrite(image_save_dir_val + "{}.{}".format(frame_idx, save_format), frame)
    with open(os.path.join(label_save_dir_val, str(frame_idx) + '.txt'), 'w') as f:
        line = create_yolo_rat_bb_line(labels[frame_idx], frame_width, frame_height, 40, 0)        
        f.write(line)
        f.write('\n')

print("Data saved to: ", output_dir)