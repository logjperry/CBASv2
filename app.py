import eel
import time
import ctypes
import yaml
import os
import math
import sys
from sys import exit
import subprocess
import shutil


eel.init('frontend')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')

@eel.expose
def project_exists(project_directory):
    project = project_directory

    cameras = os.path.join(project, 'cameras')
    recordings = os.path.join(project, 'recordings')
    models = os.path.join(project, 'models')
    data_sets = os.path.join(project, 'data_sets')

    if os.path.exists(project):
        if os.path.exists(cameras) and os.path.exists(recordings) and os.path.exists(models) and os.path.exists(data_sets):
            return True
        else:
            return False
    else:
        return False

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
        raise Exception('Project already exists. Please choose a different name or location.')
    else:
        print('Creating project...')

    # make all those directories
    os.mkdir(project)
    os.mkdir(cameras)
    os.mkdir(recordings)
    os.mkdir(models)
    os.mkdir(data_sets)

    print(f'Project creation successful!')

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
def remove_camera(camera_directory, name):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):
        print('Removing camera...')
        shutil.rmtree(camera)
    else:
        raise Exception('The camera does not exist. Please create the camera first before removing it.')

eel.start('frontend/index.html', mode='electron')
