import eel
import time
import ctypes
import yaml
import os
import math
import sys
import random
from sys import exit
import subprocess
import shutil

import cv2

import base64
from datetime import datetime, timezone

import h5py

import torch
from torch import nn 
from transformers import AutoImageProcessor, AutoModel

from decord import VideoReader
from decord import cpu, gpu

class encoder(nn.Module):

    def __init__(self, device='cuda'):
        super(encoder, self).__init__()

        self.device = device

        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def forward(self, x):

        B, S, H, W = x.shape

        x = x.to(self.device)

        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        x = x.reshape(B*S, 3, H, W)

        inputs = self.processor(images=x, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        cls = out.last_hidden_state[:, 0, :].reshape(B, S, 768)

        return cls


active_streams = {}

recordings = ''

eel.init('frontend')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')

@eel.expose
def make_recording_dir(root, sub_dir, camera_name):

    if sub_dir=='':
        sub_dir = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(os.path.join(root, sub_dir)):
        os.mkdir(os.path.join(root, sub_dir))
    
    cam_session = camera_name + '_' + datetime.now().strftime('%I%M%S-%p')    
    
    if not os.path.exists(os.path.join(root, sub_dir, cam_session)):
        os.mkdir(os.path.join(root, sub_dir, cam_session))

        return os.path.join(root, sub_dir, cam_session)
    
    return False

@eel.expose
def project_exists(project_directory):
    global recordings

    project = project_directory

    cameras = os.path.join(project, 'cameras')
    recordings = os.path.join(project, 'recordings')
    models = os.path.join(project, 'models')
    data_sets = os.path.join(project, 'data_sets')

    if os.path.exists(project):
        if os.path.exists(cameras) and os.path.exists(recordings) and os.path.exists(models) and os.path.exists(data_sets):
            return True, {'project':project, 'cameras':cameras, 'recordings':recordings, 'models':models, 'data_sets':data_sets}
        else:
            return False, None
    else:
        return False, None

@eel.expose
def create_project(parent_directory, project_name):

    # main project directory
    project = os.path.join(parent_directory, project_name)

    # make the names of the directories
    cameras = os.path.join(project, 'cameras')
    recordings = os.path.join(project, 'recordings')
    models = os.path.join(project, 'models')
    data_sets = os.path.join(project, 'data_sets')

    # check to see if the project already exists
    if os.path.exists(project):
        print('Project already exists. Please choose a different name or location.')
        return False, None
    else:
        print('Creating project...')

    # make all those directories
    os.mkdir(project)
    os.mkdir(cameras)
    os.mkdir(recordings)
    os.mkdir(models)
    os.mkdir(data_sets)

    print(f'Project creation successful!')

    return True, {'project':project, 'cameras':cameras, 'recordings':recordings, 'models':models, 'data_sets':data_sets}

@eel.expose
def ping_cameras(camera_directory):
    names = []
    cameras = {}

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):
            config = os.path.join(camera_directory, camera, 'config.yaml')

            with open(config, 'r') as file:
                cconfig = yaml.safe_load(file)

            names.append(cconfig['name'])

            print(f'Loading camera: {cconfig["name"]} at {cconfig["rtsp_url"]}...')
            
            rtsp_url = cconfig['rtsp_url']

            frame_location = os.path.join(camera_directory, camera, 'frame.jpg')

            command = f"ffmpeg -loglevel panic -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
            
            #command = f"ffmpeg -loglevel panic -rtsp_transport tcp -i {rtsp_url} -filter_complex \"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale={scale}:{scale},select=eq(n\,34)[cropped]\" -map [cropped] -vframes 1 -y {preview_location}"
        
            subprocess.Popen(command, shell=True)

            print(f'Finished loading camera: {cconfig["name"]} at {cconfig["rtsp_url"]}...')

            cameras[camera] = frame_location

    return names, cameras

@eel.expose
def update_camera_frames(camera_directory):

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):

            frame_location = os.path.join(camera_directory, camera, 'frame.jpg')
            if os.path.exists(frame_location):
                frame = cv2.imread(frame_location)

                ret, frame = cv2.imencode('.jpg', frame)

                frame = frame.tobytes()

                blob = base64.b64encode(frame)
                blob = blob.decode("utf-8")
                
                eel.updateImageSrc(camera, blob)()
    
@eel.expose
def camera_names(camera_directory):

    names = []

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):

            names.append(camera)

    return names

@eel.expose
def create_camera(camera_directory, name, rtsp_url, framerate=10, resolution=256, crop_left_x=0, crop_top_y=0, crop_width=1, crop_height=1):
    
    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    print('Creating camera...')
    os.mkdir(camera)

    # set up the camera config
    camera_config = {
        'name':name,
        'rtsp_url':rtsp_url,
        'framerate':framerate,
        'resolution':resolution,
        'crop_left_x':crop_left_x,
        'crop_top_y':crop_top_y,
        'crop_width':crop_width,
        'crop_height':crop_height
    }

    # save the camera config
    with open(os.path.join(camera, 'config.yaml'), 'w+') as file:
        yaml.dump(camera_config, file, allow_unicode=True)

    return True, name, camera_config

@eel.expose
def update_camera(camera_directory, name, rtsp_url, framerate=10, resolution=256, crop_left_x=0, crop_top_y=0, crop_width=1, crop_height=1):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):
        # must be an update
        print('Updating camera...')

        # find the camera config
        camera_config = os.path.join(camera, 'config.yaml')

        if os.path.exists(camera_config):
            # load the camera config
            with open(camera_config, 'r') as file:
                cconfig = yaml.safe_load(file)

            cconfig['name'] = name
            cconfig['rtsp_url'] = rtsp_url
            cconfig['framerate'] = framerate
            cconfig['resolution'] = resolution
            cconfig['crop_left_x'] = crop_left_x
            cconfig['crop_top_y'] = crop_top_y
            cconfig['crop_width'] = crop_width
            cconfig['crop_height'] = crop_height

            # save the camera config
            with open(camera_config, 'w+') as file:
                yaml.dump(cconfig, file, allow_unicode=True)
        else:
            # remove the camera directory and start fresh
            shutil.rmtree(camera)
            create_camera(camera_directory, name, rtsp_url, framerate, resolution, crop_left_x, crop_top_y, crop_width, crop_height)

    else:
        # must be a new camera
        create_camera(camera_directory, name, rtsp_url, framerate, resolution, crop_left_x, crop_top_y, crop_width, crop_height)

@eel.expose
def test_camera(camera_directory, name, rtsp_url):
    
    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):

        test_frame = os.path.join(camera_directory, name, 'frame.jpg')

        command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
        try:
            subprocess.call(command, shell=True)
            print('RTSP functional!')
        except:
            raise Exception('Either the RTSP url is wrong or ffmpeg is not installed properly. You may need to include a username and password in your RTSP url if you see an authorization error. You may run this function again with safe=False to generate the camera regardless of image acquisition.')

    else:
        raise Exception('The camera does not exist. Please create the camera first before testing it.')

@eel.expose
def get_cam_settings(camera_directory, name):
    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):

        # find the camera config
        camera_config = os.path.join(camera, 'config.yaml')

        with open(camera_config, 'r') as file:
            cconfig = yaml.safe_load(file)

        return cconfig
    
    return False

@eel.expose
def remove_camera(camera_directory, name):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):
        print('Removing camera...')
        shutil.rmtree(camera)
    else:
        raise Exception('The camera does not exist. Please create the camera first before removing it.')

@eel.expose
def start_camera_stream(camera_directory, name, destination, segment_time, duration=None):

    global active_streams

    if name in active_streams.keys():
        return False

    config = os.path.join(camera_directory, name, 'config.yaml')

    with open(config, 'r') as file:
        cconfig = yaml.safe_load(file)
    
    rtsp_url = cconfig['rtsp_url']
    framerate = cconfig['framerate']
    scale = cconfig['resolution']

    cw = cconfig['crop_width']
    ch = cconfig['crop_height']

    cx = cconfig['crop_left_x']
    cy = cconfig['crop_top_y']

    if not os.path.exists(destination):
        os.mkdir(destination)

    destination = os.path.join(destination, f'{name}_%05d.mp4')

    command = [
        'ffmpeg', '-loglevel', 'panic', '-rtsp_transport', 'tcp', '-i', str(rtsp_url), 
        '-r', str(framerate), 
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale={scale}:{scale}[cropped]", 
        '-map', '[cropped]', '-f', 'segment', '-segment_time', str(segment_time), 
        '-reset_timestamps', '1',
        '-hls_flags', 'temp_file', '-y', destination
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

    active_streams[name] = process

    return True

@eel.expose
def stop_camera_stream(camera_name):
    global active_streams

    if camera_name in active_streams.keys():
        active_streams[camera_name].communicate(input=b'q')
        active_streams.pop(camera_name)

        return True 
    
    return False

@eel.expose
def kill_streams():
    global active_streams

    for stream in active_streams.keys():
        active_streams[stream].communicate(input=b'q')

eel.start('frontend/index.html', mode='electron', block=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = encoder(device).to(device)

while True:
    
    if recordings!='':

        sub_dirs = [os.path.join(recordings, d) for d in os.listdir(recordings) if os.path.isdir(os.path.join(recordings, d))]

        for sd in sub_dirs:

            sub_sub_dirs = [os.path.join(sd, d) for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))]

            for ssd in sub_sub_dirs:

                videos = [os.path.join(ssd, vid) for vid in os.listdir(ssd) if vid.endswith('.mp4') and not os.path.exists(os.path.join(ssd, vid.replace('.mp4', '_cls.h5')))]

                for v in videos:

                    try:

                        cap = cv2.VideoCapture(v)

                        if not cap.isOpened():
                            continue

                        vr = VideoReader(v, ctx=cpu(0))
                        
                        frames = vr.get_batch(range(0, len(vr), 1)).asnumpy()

                        frames = torch.from_numpy(frames[:, :, :, 1]).float()

                        batch_size = 256

                        clss = []

                        for i in range(0, len(frames), batch_size):
                            batch = frames[i:i+batch_size]

                            with torch.no_grad():
                                out = enc(batch.unsqueeze(1).to(device))

                            out = out.squeeze(1).to('cpu')
                            clss.extend(out)

                        file_path = os.path.splitext(v)[0]+'_cls.h5'

                        with h5py.File(file_path, 'w') as file:
                            file.create_dataset('cls', data=torch.stack(clss).numpy())
                        
                    except Exception as e:
                        print('Error processing video:', v)
                        continue

        eel.sleep(5.0)
    else:
        eel.sleep(60.0) 
