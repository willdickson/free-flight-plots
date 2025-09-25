import os
import vtk
import pickle
import collections
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import time
import math
import cv2
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.special
from scipy import interpolate

# Local package imports
from .body_class import BodyModel
from .wing_twist_class import WingModel_L
from .wing_twist_class import WingModel_R

class FreeFlightPlots():

    def __init__(self):
        self.str_to_axis = {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}
        self.num_to_cam_view = {1 : 'cam1', 2 : 'cam2', 3 : 'cam3'}
        self.rtool_data = collections.OrderedDict([
                ('q'     , np.array([1, 0, 0, 0])),
                ('axis'  , 'x'),
                ('angle' , 10.0),
                ('quit'  , False),
                ('frame' , {'axis': (1,0,0), 'angle': 0.0}),
                ])


    def load_data(self, filename, show=False):
        self.data_filename = filename
        data = np.loadtxt(self.data_filename)
        num_pts = data.shape[1]

        self.f = data[0,:]
        self.f -= self.f[0]
        self.t = data[1,:]
        self.x = data[2,:]
        self.y = data[3,:]
        self.z = data[4,:]

        self.vx = data[5,:]
        self.vy = data[6,:]
        self.vz = data[7,:]

        self.qx = data[8,:]
        self.qy = data[9,:]
        self.qz = data[10,:]
        self.q0 = data[11,:]

        self.wx = data[12,:]
        self.wy = data[13,:]
        self.wz = data[14,:]

        self.phi_L = data[15,:]
        self.phi_R = data[16,:]

        self.theta_L = data[17,:]
        self.theta_R = data[18,:]

        self.eta_L = data[19,:]
        self.eta_R = data[20,:]
        
        if show:
            self.plot_kinematics()

        self.phi_L = np.deg2rad(self.phi_L)
        self.phi_R = np.deg2rad(self.phi_R)

        self.theta_L = np.deg2rad(self.theta_L)
        self.theta_R = np.deg2rad(self.theta_R)

        self.eta_L = np.deg2rad(self.eta_L)
        self.eta_R = np.deg2rad(self.eta_R)
        self.zero_frame()



    def plot_kinematics(self):

        if 0:
            fig, ax = plt.subplots(1,1,sharex=True)
            ax.plot(self.f,self.x, 'r')
            ax.plot(self.f,self.y, 'g')
            ax.plot(self.f,self.z, 'b')
            ax.set_ylabel('pos')
            ax.grid(True)
            ax.set_ylim(-40,40)
            ax.set_xlabel('frame')
            plt.show()

        if 1:
            fig, ax = plt.subplots(7,1,sharex=True)
            ax[0].plot(self.f,self.x, 'r')
            ax[0].plot(self.f,self.y, 'g')
            ax[0].plot(self.f,self.z, 'b')
            ax[0].set_ylabel('pos')
            ax[0].grid(True)
            ax[0].set_ylim(-40,40)
            
            ax[1].plot(self.f,self.vx, 'r')
            ax[1].plot(self.f,self.vy, 'g')
            ax[1].plot(self.f,self.vz, 'b')
            ax[1].set_ylabel('vel')
            ax[1].grid(True)
            ax[1].set_ylim(-500,500)
            
            ax[2].plot(self.f,self.q0, 'k')
            ax[2].plot(self.f,self.qx, 'r')
            ax[2].plot(self.f,self.qy, 'g')
            ax[2].plot(self.f,self.qz, 'b')
            ax[2].set_ylabel('quat')
            ax[2].grid(True)
            ax[2].set_ylim(-1,1)
            
            ax[3].plot(self.f,self.wx, 'r')
            ax[3].plot(self.f,self.wy, 'g')
            ax[3].plot(self.f,self.wz, 'b')
            ax[3].set_ylabel('angvel')
            ax[3].grid(True)
            ax[3].set_ylim(-70,90)
            
            ax[4].plot(self.f,self.phi_R, 'r')
            ax[4].plot(self.f,self.phi_L, 'b')
            ax[4].set_ylabel('phi')
            ax[4].grid(True)
            ax[4].set_ylim(-80,80)
            
            ax[5].plot(self.f,self.theta_R, 'r')
            ax[5].plot(self.f,self.theta_L, 'b')
            ax[5].set_ylabel('theta')
            ax[5].grid(True)
            ax[5].set_ylim(-30,30)
            
            ax[6].plot(self.f,self.eta_R, 'r')
            ax[6].plot(self.f,self.eta_L, 'b')
            ax[6].set_ylabel('eta')
            ax[6].grid(True)
            ax[6].set_ylim(-80,80)
            
            ax[6].set_xlabel('frame')
            plt.show()


    def Renderer(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)


    def ConstructModel(self):
        self.body_mdl = BodyModel()
        self.body_mdl.add_actors(self.ren)
        self.wing_mdl_L = WingModel_L()
        self.wing_mdl_L.add_actors(self.ren)
        self.wing_mdl_R = WingModel_R()
        self.wing_mdl_R.add_actors(self.ren)


    def ScaleModel(self,scale_in):
        self.scale = scale_in
        self.body_mdl.scale_head(scale_in[0])
        self.body_mdl.scale_thorax(scale_in[1])
        self.body_mdl.scale_abdomen(scale_in[2])
        self.wing_mdl_L.scale_wing(scale_in[3])
        self.wing_mdl_R.scale_wing(scale_in[4])


    def SetModelState(self,s_head_in,s_thorax_in,s_abdomen_in,s_wing_L_in,s_wing_R_in):
        # head
        self.body_mdl.transform_head(s_head_in)
        self.s_head = np.copy(s_head_in)
        # thorax
        self.body_mdl.transform_thorax(s_thorax_in)
        self.s_thorax = np.copy(s_thorax_in)
        # abdomen
        self.body_mdl.transform_abdomen(s_abdomen_in)
        self.s_abdomen = np.copy(s_abdomen_in)
        # wing L
        self.wing_mdl_L.transform_wing(s_wing_L_in)
        self.s_wing_L = np.copy(s_wing_L_in)
        # wing R
        self.wing_mdl_R.transform_wing(s_wing_R_in)
        self.s_wing_R = np.copy(s_wing_R_in)


    def q_mult(self,qA,qB,normalize=True):
        QA = np.array([
            [qA[0],  -qA[1],  -qA[2],  -qA[3]],
            [qA[1],   qA[0],  -qA[3],   qA[2]],
            [qA[2],   qA[3],   qA[0],  -qA[1]],
            [qA[3],  -qA[2],   qA[1],   qA[0]]
            ])
        qC = np.dot(QA,qB)
        if normalize:
            qC_norm = math.sqrt(pow(qC[0],2)+pow(qC[1],2)+pow(qC[2],2)+pow(qC[3],2))
            if qC_norm>0.01:
                qC /= qC_norm
            else:
                qC = np.array([1.0,0.0,0.0,0.0])
        return qC


    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        M = np.array([
            [2*pow(q0,2)-1+2*pow(q1,2),            2*q1*q2-2*q0*q3,                     2*q1*q3+2*q0*q2,   tx],
            [          2*q1*q2+2*q0*q3,  2*pow(q0,2)-1+2*pow(q2,2),           2*q2*q3-2*q0*q1,             ty],
            [          2*q1*q3-2*q0*q2,            2*q2*q3+2*q0*q1,           2*pow(q0,2)-1+2*pow(q3,2),   tz],
            [                        0,                0,                                             0,    1]
            ])
        return M

    def zero_frame(self):
        self.x -= self.x[0] 
        self.y -= self.y[0]
        self.z -= self.z[0]


    def rotate_frame(self, axis, angle):
        ax, ay, az = axis
        num_pts = len(self.x)
        q_rot = axis_angle_to_quat(axis, angle)
        q_inv = quat_inverse(q_rot)
        for i in range(num_pts):
            p = (0, self.x[i], self.y[i], self.z[i])
            p = self.q_mult(q_rot, self.q_mult(p, q_inv, normalize=False), normalize=False)
            _, px, py, pz = p
            self.x[i] = px 
            self.y[i] = py
            self.z[i] = pz 
        for i in range(num_pts):
            q = self.q0[i], self.qx[i], self.qy[i], self.qz[i]
            q = self.q_mult(q_rot, q)
            self.q0[i] = q[0]
            self.qx[i] = q[1]
            self.qy[i] = q[2]
            self.qz[i] = q[3]
        self.rtool_data['frame'] = {'axis': axis, 'angle': angle}


    def rotate_axis_by_quat(self, axis, q):
        ax, ay, az = axis
        p = 0, ax, ay, az
        q_inv = q[0], -q[1], -q[2], -q[3]
        p = self.q_mult(q, self.q_mult(p, q_inv))
        _, ax, ay, az = p
        return ax, ay, az


    def set_srf_angle(self,beta_in):
        self.beta = beta_in


    def set_wing_motion(self): 
        x_L = np.zeros((7,))
        x_R = np.zeros((7,))
        N_points = self.t.shape[0]
        self.state_mat_L = np.zeros((8, N_points))
        self.state_mat_R = np.zeros((8, N_points))
        for j in range(N_points):
            x_L[0] = self.phi_L[j]
            x_L[1] = self.theta_L[j]
            x_L[2] = self.eta_L[j]
            x_L[3] = 0.0
            x_L[4] = 0.0
            x_L[5] = 0.6
            x_L[6] = 0.0
            x_R[0] = self.phi_R[j]
            x_R[1] = self.theta_R[j]
            x_R[2] = self.eta_R[j]
            x_R[3] = 0.0 
            x_R[4] = 0.0
            x_R[5] = -0.6
            x_R[6] = 0.0
            self.state_mat_L[:,j] = self.calculate_state_L(x_L)
            self.state_mat_R[:,j] = self.calculate_state_R(x_R)


    def calculate_state_L(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        #q_start = np.array([1.0,0.0,0.0,0.0])
        q_phi = np.array([np.cos(-phi/2.0),np.sin(-phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(theta/2.0),0.0,0.0,np.sin(theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_L = self.q_mult(q_eta,self.q_mult(q_theta,self.q_mult(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_L[0]
        state_out[1] = q_L[1]
        state_out[2] = q_L[2]
        state_out[3] = q_L[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out


    def calculate_state_R(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        #q_start = np.array([1.0,0.0,0.0,0.0])
        q_phi = np.array([np.cos(phi/2.0),np.sin(phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(-theta/2.0),0.0,0.0,np.sin(-theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_R = self.q_mult(q_eta,self.q_mult(q_theta,self.q_mult(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_R[0]
        state_out[1] = q_R[1]
        state_out[2] = q_R[2]
        state_out[3] = q_R[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out


    def make_video_moving(self,filename, qinit, scale_in=20, img_w=1000, img_h=800, view_list=[], step_size=1):
        self.initialize_orientation(qinit)
        N = self.t.shape[0]
        self.s_body = np.zeros((7,N))
        self.s_body[0,:] = self.q0
        self.s_body[1,:] = self.qx
        self.s_body[2,:] = self.qy
        self.s_body[3,:] = self.qz
        self.s_body[4,:] = self.x
        self.s_body[5,:] = self.y
        self.s_body[6,:] = self.z
        self.set_wing_motion()
        self.create_video(filename, scale_in, img_w, img_h, view_list, step_size)


    def make_video_stationary(self, filename, qinit, scale_in=3, img_w=1000, img_h=800, view_list=[], step_size=1):
        #self.initialize_orientation(qinit)
        self.zero_orientation()
        N = self.t.shape[0]
        self.s_body = np.zeros((7,N))
        self.s_body[0,:] = self.q0[0]
        self.s_body[1,:] = self.qx[0]
        self.s_body[2,:] = self.qy[0]
        self.s_body[3,:] = self.qz[0]
        self.s_body[4,:] = 0.0
        self.s_body[5,:] = 0.0
        self.s_body[6,:] = 0.0
        self.set_wing_motion()
        self.create_video(filename, scale_in, img_w, img_h, view_list, step_size)


    def zero_orientation(self):
        qinit_inv = quat_inverse((self.q0[0], self.qx[0], self.qy[0], self.qz[0]))
        self.initialize_orientation(qinit_inv)


    def initialize_orientation(self, qinit):
        num_pts = self.t.shape[0]
        for i in range(num_pts):
            q = self.q0[i], self.qx[i], self.qy[i], self.qz[i]
            q = self.q_mult(q, qinit)
            self.q0[i] = q[0]
            self.qx[i] = q[1]
            self.qy[i] = q[2]
            self.qz[i] = q[3]

    def create_video(self, filename, scale_in, img_w, img_h, view_list, step_size=1):
        s_t = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,1.0])
        s_h = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.6*0.9,0.0,0.42*0.9,1.0])
        s_a = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1*0.9,1.0])

        self.body_mdl.set_Color([(0.5,0.5,0.5)])
        self.wing_mdl_L.set_Color([(1.0,0.0,0.0),(1.0,0.0,0.0),0.3])
        self.wing_mdl_R.set_Color([(0.0,0.0,1.0),(0.0,0.0,1.0),0.3])

        self.wing_mdl_L.transform_wing(self.state_mat_L[:,0])
        self.wing_mdl_R.transform_wing(self.state_mat_R[:,0])
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(img_w, img_h)

        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        self.set_camera_view1(camera, scale_in)
        self.renWin.Render()

        N_steps = self.state_mat_L.shape[1]
        s_head    = np.zeros(7)
        s_thorax  = np.zeros(7)
        s_abdomen = np.zeros(7)
        x_wing_L  = np.ones(4)
        s_wing_L  = np.zeros(8)
        x_wing_R  = np.ones(4)
        s_wing_R  = np.zeros(8)
        
        if not view_list:
            view_list = ['cam1', 'cam2', 'cam3']

        out = None
        for i in range(0,N_steps-1,step_size): 
            print(f'{i}/{N_steps-1}')
            M_body        = self.quat_mat(self.s_body[:,i])
            q_head        = self.q_mult(self.s_body[:4,i],s_h[:4])
            p_head        = np.dot(M_body,s_h[4:])
            s_head[:4]    = q_head
            s_head[4:]    = p_head[:3]
            q_thorax      = self.q_mult(self.s_body[:4,i],s_t[:4])
            p_thorax      = np.dot(M_body,s_t[4:])
            s_thorax[:4]  = q_thorax
            s_thorax[4:]  = p_thorax[:3]
            q_abdomen     = self.q_mult(self.s_body[:4,i],s_a[:4])
            p_abdomen     = np.dot(M_body,s_a[4:])
            s_abdomen[:4] = q_abdomen
            s_abdomen[4:] = p_abdomen[:3]
            q_body        = self.s_body[:4,i]
            q_body[1]     = -q_body[1]
            q_body[2]     = -q_body[2]
            q_body[3]     = -q_body[3]
            q_wing_L      = self.q_mult(self.state_mat_L[:4,i],q_body)
            x_wing_L[:3]  = self.state_mat_L[4:7,i]
            p_wing_L      = np.dot(M_body,x_wing_L)
            s_wing_L[:4]  = q_wing_L
            s_wing_L[4:7] = p_wing_L[:3]
            s_wing_L[7]   = self.state_mat_L[7,i]
            q_wing_R      = self.q_mult(self.state_mat_R[:4,i],q_body)
            x_wing_R[:3]  = self.state_mat_R[4:7,i]
            p_wing_R      = np.dot(M_body,x_wing_R)
            s_wing_R[:4]  = q_wing_R
            s_wing_R[4:7] = p_wing_R[:3]
            s_wing_R[7]   = self.state_mat_R[7,i]

            self.body_mdl.transform_head(s_head)
            self.body_mdl.transform_thorax(s_thorax)
            self.body_mdl.transform_abdomen(s_abdomen)
            self.wing_mdl_L.transform_wing(s_wing_L)
            self.wing_mdl_R.transform_wing(s_wing_R)
            self.renWin.Render()

            img_list = []
            for cam_view in view_list:

                self.set_camera_view(cam_view, camera, scale_in)
                self.renWin.Render()

                w2i = vtk.vtkWindowToImageFilter()
                w2i.SetInput(self.renWin)
                w2i.SetInputBufferTypeToRGB()
                w2i.ReadFrontBufferOff()
                w2i.Update()
                img_i = w2i.GetOutput()

                n_rows, n_cols, _ = img_i.GetDimensions()
                img_sc = img_i.GetPointData().GetScalars()
                np_img = vtk_to_numpy(img_sc)
                np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
                cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                img_list.append(cv_img)

            big_img = np.hstack(img_list)
            if out is None:
                size = (big_img.shape[1], big_img.shape[0])
                out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
            if i > 1:
                out.write(big_img)
        out.release()

        # Reset body and wing models
        tip_start_L = np.array([-1.8,0.5,1.8])
        tip_start_R = np.array([-1.8,-0.5,1.8])
        self.wing_mdl_L.clear_root_tip_pts(np.zeros(3),tip_start_L[:3])
        self.wing_mdl_R.clear_root_tip_pts(np.zeros(3),tip_start_R[:3])
        self.renWin.Render()


    def rotation_tool(self, nstop=200, cam_view_in ='cam3', scale_in=20):
        self.rtool_data['view']  = cam_view_in
        self.rtool_data['scale'] = scale_in
        img_w = 1000 
        img_h = 800 

        N = self.t.shape[0]
        self.s_body = np.zeros((7,N))
        self.s_body[0,:] = self.q0
        self.s_body[1,:] = self.qx
        self.s_body[2,:] = self.qy
        self.s_body[3,:] = self.qz
        self.s_body[4,:] = self.x
        self.s_body[5,:] = self.y
        self.s_body[6,:] = self.z
        self.set_wing_motion()

        s_t = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,1.0])
        s_h = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.6*0.9,0.0,0.42*0.9,1.0])
        s_a = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1*0.9,1.0])

        self.body_mdl.set_Color([(0.5,0.5,0.5)])
        self.wing_mdl_L.set_Color([(1.0,0.0,0.0),(1.0,0.0,0.0),0.3])
        self.wing_mdl_R.set_Color([(0.0,0.0,1.0),(0.0,0.0,1.0),0.3])

        self.wing_mdl_L.transform_wing(self.state_mat_L[:,0])
        self.wing_mdl_R.transform_wing(self.state_mat_R[:,0])
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(img_w,img_h)

        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)

        cam_view = self.rtool_data['view']
        scale = self.rtool_data['scale']
        self.set_camera_view(cam_view, camera, scale_in)

        self.renWin.Render()
        N_steps = self.state_mat_L.shape[1]

        s_head    = np.zeros(7)
        s_thorax  = np.zeros(7)
        s_abdomen = np.zeros(7)
        x_wing_L  = np.ones(4)
        s_wing_L  = np.zeros(8)
        x_wing_R  = np.ones(4)
        s_wing_R  = np.zeros(8)

        i = -1 
        while True:
            if i > nstop:
                self.rotation_tool_options()
                try:
                    nstep = self.rtool_data['step']
                except KeyError:
                    pass
                else:
                    del self.rtool_data['step']
                    nstop += nstep
            else:
                i += 1
                print(f'{i}/{N_steps-1}')

            self.s_body[0,:] = self.q0
            self.s_body[1,:] = self.qx
            self.s_body[2,:] = self.qy
            self.s_body[3,:] = self.qz
            self.s_body[4,:] = self.x
            self.s_body[5,:] = self.y
            self.s_body[6,:] = self.z
            cam_view = self.rtool_data['view']
            scale = self.rtool_data['scale']
            self.set_camera_view(cam_view, camera, scale)

            if self.rtool_data['quit']:
                print('quiting')
                print()
                break

            M_body        = self.quat_mat(self.s_body[:,i])
            q_head        = self.q_mult(self.s_body[:4,i],s_h[:4])
            p_head        = np.dot(M_body,s_h[4:])
            s_head[:4]    = q_head
            s_head[4:]    = p_head[:3]
            q_thorax      = self.q_mult(self.s_body[:4,i],s_t[:4])
            p_thorax      = np.dot(M_body,s_t[4:])
            s_thorax[:4]  = q_thorax
            s_thorax[4:]  = p_thorax[:3]
            q_abdomen     = self.q_mult(self.s_body[:4,i],s_a[:4])
            p_abdomen     = np.dot(M_body,s_a[4:])
            s_abdomen[:4] = q_abdomen
            s_abdomen[4:] = p_abdomen[:3]
            q_body        = self.s_body[:4,i]
            q_body[1]     = -q_body[1]
            q_body[2]     = -q_body[2]
            q_body[3]     = -q_body[3]
            q_wing_L      = self.q_mult(self.state_mat_L[:4,i],q_body)
            x_wing_L[:3]  = self.state_mat_L[4:7,i]
            p_wing_L      = np.dot(M_body,x_wing_L)
            s_wing_L[:4]  = q_wing_L
            s_wing_L[4:7] = p_wing_L[:3]
            s_wing_L[7]   = self.state_mat_L[7,i]
            q_wing_R      = self.q_mult(self.state_mat_R[:4,i],q_body)
            x_wing_R[:3]  = self.state_mat_R[4:7,i]
            p_wing_R      = np.dot(M_body,x_wing_R)
            s_wing_R[:4]  = q_wing_R
            s_wing_R[4:7] = p_wing_R[:3]
            s_wing_R[7]   = self.state_mat_R[7,i]
            self.body_mdl.transform_head(s_head)
            self.body_mdl.transform_thorax(s_thorax)
            self.body_mdl.transform_abdomen(s_abdomen)
            self.wing_mdl_L.transform_wing(s_wing_L)
            self.wing_mdl_R.transform_wing(s_wing_R)
            self.renWin.Render()

            #Export a single frame
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.renWin)
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()
            #img_i = w2i.GetOutput()
            #n_rows, n_cols, _ = img_i.GetDimensions()
            #img_sc = img_i.GetPointData().GetScalars()
            #np_img = vtk_to_numpy(img_sc)
            #np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
            #cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
            #out.write(cv_img)

        #out.release()

        # Reset body and wing models
        tip_start_L = np.array([-1.8,0.5,1.8])
        tip_start_R = np.array([-1.8,-0.5,1.8])

        self.wing_mdl_L.clear_root_tip_pts(np.zeros(3),tip_start_L[:3])
        self.wing_mdl_R.clear_root_tip_pts(np.zeros(3),tip_start_R[:3])
        self.renWin.Render()
        
    def set_camera_view(self, cam_view, camera, scale_in):
        match cam_view:
            case 'cam1':
                self.set_camera_view1(camera, scale_in)
            case 'cam2':
                self.set_camera_view2(camera, scale_in)
            case 'cam3':
                self.set_camera_view3(camera, scale_in)
            case 'cam4':
                self.set_camera_view4(camera, scale_in)
            case 'cam5':
                self.set_camera_view5(camera, scale_in)
            case _:
                print(f'unknown camera: {cam_view}')

    def set_camera_view1(self, camera, scale_in):
        camera.SetParallelScale(scale_in)
        camera.SetPosition(-50.0, 0.0, 0.0)
        camera.SetClippingRange(0.0,200.0)
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(0.0,0.0,1.0)
        camera.OrthogonalizeViewUp()

    def set_camera_view2(self, camera, scale_in):
        camera.SetParallelScale(scale_in)
        camera.SetPosition(0.0, -50.0, 0.0)
        camera.SetClippingRange(0.0,200.0)
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(0.0,0.0,1.0)
        camera.OrthogonalizeViewUp()

    def set_camera_view3(self, camera, scale_in):
        camera.SetParallelScale(scale_in)
        camera.SetPosition(0.0, 0.0, 100.0)
        camera.SetClippingRange(0.0,500.0)
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(1.0,0.0,0.0)
        camera.OrthogonalizeViewUp()

    def set_camera_view4(self, camera, scale_in):
        camera.SetParallelScale(scale_in)
        camera.SetPosition(50.0, 50.0, 20.0)
        camera.SetClippingRange(0.0,200.0)
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(0.0,0.0,1.0)
        camera.OrthogonalizeViewUp()

    def set_camera_view5(self, camera, scale_in):
        camera.SetParallelScale(scale_in)
        camera.SetPosition(-50.0, 30.0, 0.0)
        camera.SetClippingRange(0.0,200.0)
        camera.SetFocalPoint(0.0,10.0,0.0)
        camera.SetViewUp(0.0,0.0,1.0)
        camera.OrthogonalizeViewUp()

    def rotation_tool_options(self):
        option = input('>> ')
        option = [x for x in option.split(' ') if x]
        match option:
            case 'h' | 'help',:
                self.show_rotation_tool_help()
            case 'r', axis: 
                self.set_rotation_tool_axis(axis)
            case 'a', angle:
                self.set_rotation_tool_angle(angle)
            case 'f',:
                self.rotation_tool_forward_step()
            case 'b',:
                self.rotation_tool_backward_step()
            case 'v', nview:
                self.rotation_tool_set_view(nview)
            case 'z', scale:
                self.rotation_tool_set_scale(scale)
            case 's',:
                self.rotation_tool_step(1)
            case 's', nstep:
                self.rotation_tool_step(nstep)
            case 'o',:
                self.rotation_tool_save()
            case 'i',:
                self.show_rotation_tool_info()
            case 'q' | 'quit',:
                self.rtool_data['quit'] = True
            case _:
                print('unknown command')

    def show_rotation_tool_help(self):
        print()
        print('options')
        print('-'*60)
        print(' r n   - select n-axis as rotation axis')
        print(' a n   - set rotation angle to n')
        print(' f     - step rotation forward')
        print(' b     - step rotation backward')
        print(' v n   - set camera view 1,2 or 3')
        print(' z n   - set view zoom/scale factor') 
        print(' s n   - step simulation')
        print(' o     - quaternion to file')
        print(' i     - show rotation tool info')
        print(' q     - quit')
        print()

    def set_rotation_tool_axis(self, axis):
        if not axis in ('x', 'y', 'z'):
            print(f'unknown rotation axis {axis}')
        else:
            print(f'setting rotation axis to {axis}')
            self.rtool_data['axis'] = axis

    def set_rotation_tool_angle(self, angle):
        try:
            fangle = float(angle)
        except ValueError:
            print(f'unable to convert angle {angle} to float')
        else:
            print(f'setting rotation angle to {fangle}')
            self.rtool_data['angle'] = fangle

    def rotation_tool_forward_step(self):
        angle = self.rtool_data['angle']
        axis_str = self.rtool_data['axis']
        print(f'rotation by {angle}(deg) about {axis_str}-axis')
        axis = self.str_to_axis[axis_str]
        self.rotate_quat_array(axis, np.deg2rad(angle))

    def rotation_tool_backward_step(self):
        angle = -self.rtool_data['angle']
        axis_str = self.rtool_data['axis']
        print(f'rotation by {angle}(deg) about {axis_str}-axis')
        axis = self.str_to_axis[axis_str]
        self.rotate_quat_array(axis, np.deg2rad(angle))

    def rotation_tool_save(self):
        init_data = {
                'model_qinit'    : self.rtool_data['q'],
                'frame_rotation' : {
                    'axis'  : self.rtool_data['frame']['axis'],
                    'angle' : self.rtool_data['frame']['angle'],
                    },
                }
        save_file = f'init_{self.data_filename.stem}.pkl'
        with open(save_file, 'wb') as f:
            pickle.dump(init_data,f)
        print(f'initialization info save to  {save_file}')

    def rotation_tool_set_view(self, nview):
        try:
            cam_view = self.num_to_cam_view[int(nview)]
        except (KeyError, ValueError):
            print(f'unknown camera view number {nview}')
        else:
            print(f'setting camera view to {cam_view}')
            self.rtool_data['view'] = cam_view

    def rotation_tool_set_scale(self, scale):
        try:
            fscale = float(scale)
        except ValueError:
            print(f'unable to convert zoom/scale to float')
            return
        if fscale < 0.1:
            print(f'zoom/scale {fscale} too small')
            return
        self.rtool_data['scale'] = fscale

    def rotation_tool_step(self,step):
        try:
            nstep = int(step)
        except ValueError:
            print(f'unable to convert step {step} to int')
        else:
            self.rtool_data['step'] = nstep 

    def show_rotation_tool_info(self):
        print()
        print('rotation info')
        print('-'*60)
        for k, v in self.rtool_data.items():
            print(f'{k:<10} {str(v):<10}')
        print()

    def rotate_quat_array(self, axis, angle):
        num_pts = len(self.q0)
        qstep = axis_angle_to_quat(axis, angle)
        self.rtool_data['q'] = self.q_mult(self.rtool_data['q'], qstep)
        for i in range(num_pts):
            q = self.q0[i], self.qx[i], self.qy[i], self.qz[i]
            q = self.q_mult(q, qstep)
            self.q0[i] = q[0]
            self.qx[i] = q[1]
            self.qy[i] = q[2]
            self.qz[i] = q[3]


# Utility functions
# -----------------------------------------------------------------------

def axis_angle_to_quat(axis, angle):
    ax, ay, az = axis
    q0 = np.cos(angle/2)
    qx = ax*np.sin(angle/2)
    qy = ay*np.sin(angle/2)
    qz = az*np.sin(angle/2)
    q = np.array([q0, qx, qy, qz])
    norm = np.sqrt(q0**2 + qx**2 + qy**2 + qz**2)
    if norm>0.01:
        q /= norm
    else:
        q = np.array([1.0,0.0,0.0,0.0])
    return q

def quat_inverse(q):
    return q[0], -q[1], -q[2], -q[3]


