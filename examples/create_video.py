import os
import sys
import pickle
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt
from free_flight_plots import FreeFlightPlots

init_info_filename = 'init_info.pkl'
video_type = 'moving'  # moving/stationary
step_size = 1

file_to_cam_views = collections.OrderedDict([ 
    ('20121009_S0003', ['cam3']),
    ('20121011_S0004', ['cam1']), 
    ('20121120_S0001', ['cam2']), 
    ('20121127_S0004', ['cam3']), 
    ('20121128_S0002', ['cam1']), 
    ('20121128_S0003', ['cam5']), 
    #('20121009_S0003', ['cam1', 'cam2', 'cam3']),  # example w/ three cameras
    ])

file_list = sys.argv[1:]

try:
    with open(init_info_filename, 'rb') as f:
        init_info = pickle.load(f)
except FileNotFoundError:
    print('initialization information not found')
    qinit = np.array([1.0, 0.0, 0.0, 0.0])
    frame_rotation_axis = (0,0,1)
    frame_rotation_angle = 0.0
else:
    for k,v in init_info.items():
        print(f'{k}: {v}')
    print()
    qinit = init_info['model_qinit']
    frame_rotation_axis  = init_info['frame_rotation']['axis']
    frame_rotation_angle = init_info['frame_rotation']['angle']


for item in file_list:

    data_file = pathlib.Path(item)
    video_file = f'{data_file.stem}_{video_type}_render.avi'
    print(f'data_file:  {data_file}')
    print(f'video_file: {video_file}')

    
    ffplots = FreeFlightPlots()
    ffplots.Renderer()
    ffplots.ConstructModel()

    
    # Scale model:
    mdl_scale = [0.8,0.8,0.8,1.0,1.0]
    ffplots.ScaleModel(mdl_scale)
    
    # Orient body model:
    s_thorax  = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0])
    s_head 	  = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.55*mdl_scale[0],0.0,0.42*mdl_scale[0]])
    s_abdomen = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1*mdl_scale[2]])
    s_wing_L  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    s_wing_R  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    ffplots.SetModelState(s_head,s_thorax,s_abdomen,s_wing_L,s_wing_R)
    
    # Set Strokeplane Reference Frame angle:
    ffplots.set_srf_angle(np.pi*(55.0/180.0))
    ffplots.load_data(data_file, show=False)
    view_list = file_to_cam_views[str(data_file.stem)]

    #ffplots.plot_kinematics()
    #plt.show()

    
    match video_type:
        case 'moving':
            ffplots.rotate_frame(frame_rotation_axis, frame_rotation_angle)
            ffplots.make_video_moving(video_file, qinit, scale_in=25, view_list=view_list, step_size=step_size)
        case 'stationary':
            ffplots.make_video_stationary(video_file, qinit, scale_in=3, view_list=view_list, step_size=step_size)
        case _:
            print(f'unknown video type {video_type}')
    

