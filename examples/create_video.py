import os
import sys
import pickle
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt
from free_flight_plots import FreeFlightPlots

file_list = sys.argv[1:]

for item in file_list:

    data_file = pathlib.Path(item)
    video_type = 'moving'
    #video_type = 'stationary'
    video_file = f'{data_file.stem}_{video_type}_render.avi'
    print(f'data_file:  {data_file}')
    print(f'video_file: {video_file}')
    step_size = 1

    file_to_cam_views = collections.OrderedDict([ 
        #('20121009_S0003', ['cam1', 'cam2', 'cam3']),  # all three cameras (example)
        ('20121009_S0003', ['cam3']),
        ('20121011_S0004', ['cam1']), 
        ('20121120_S0001', ['cam2']), 
        ('20121127_S0004', ['cam3']), 
        ('20121128_S0002', ['cam1']), 
        ('20121128_S0003', ['cam5']), 
        ])
    
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
    
    #init_info_filename = f'init_{data_file.stem}.pkl'
    init_info_filename = f'init_info.pkl'
    try:
        with open(init_info_filename, 'rb') as f:
            init_info = pickle.load(f)
    except FileNotFoundError:
        print('initialization information not found')
        qinit = np.array([1.0, 0.0, 0.0, 0.0])
        axis = (0,0,1)
        angle = 0.0
    else:
        for k,v in init_info.items():
            print(f'{k}: {v}')
        print()
        qinit = init_info['model_qinit']
        axis  = init_info['frame_rotation']['axis']
        angle = init_info['frame_rotation']['angle']
    
    
    match video_type:
        case 'moving':
            view_list = file_to_cam_views[str(data_file.stem)]
            ffplots.rotate_frame(axis, angle)
            ffplots.make_video_moving(video_file, qinit, scale_in=25, view_list=view_list, step_size=step_size)
        case 'stationary':
            view_list = ['cam4']
    
            print(f'frame rotation angle: {angle}')
            #ffplots.rotate_frame(axis, np.deg2rad(angle))
            ffplots.make_video_stationary(video_file, qinit, scale_in=3, view_list=view_list, step_size=step_size)
        case _:
            print(f'unknown video type {video_type}')
    






