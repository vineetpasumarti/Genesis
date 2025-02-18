#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np
import scipy as sp
import casadi as ca
import matplotlib.pyplot as plt

# import drone3d as d3d
from interp import spline_interpolant


@dataclass
class PythonMsg:
    '''
    base class for creating types and messages in python
    '''
    def __setattr__(self,key,value):
        '''
        Overloads default atribute-setting functionality
          to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally
          adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self,key):
            raise TypeError (f'Not allowed to add new field "{key}" to class {self}')
        else:
            object.__setattr__(self,key,value)



@dataclass
class RaceTrack(PythonMsg):
    '''
    base class for creating types and messages in python
    '''
    s_tag: np.array = field(default = None)
    x_gates: np.array = field(default = None)
    NumOfGates: int = 0

    center_spline: sp.interpolate.CubicSpline = field(default = None)
    xi: sp.interpolate.interp1d = field(default = None)
    xj: sp.interpolate.interp1d = field(default = None)
    xk: sp.interpolate.interp1d = field(default = None)

    f_xc: ca.Function = field(default = None)
    f_xcs: ca.Function = field(default = None)
    f_xcss: ca.Function = field(default = None)
    f_xcsss: ca.Function = field(default = None)

    gate_size: float = 1.2 # Size of the gate
    gate_type: str = 'square' # Type of gate (square, circle, etc.)

    sL: np.array = field(default = None)
    L: float = 0.0

    sym_s: ca.SX = field(default = None)

    f_T: ca.Function = field(default = None)
    f_B: ca.Function = field(default = None)
    f_N: ca.Function = field(default = None)
    f_k: ca.Function = field(default = None)
    f_tau: ca.Function = field(default = None)
    f_gate_factor: ca.Function = field(default = None)

    AirDensity: float = field(default = 1.225)
    gravity: float = field(default = 9.81)

    s_prev: float = 0.0


    def __init__(self, Traj: int):

        match (Traj):
            case 1:
                x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8, -1.1]).reshape(-1,1)
                y = np.array([-1.6, 06.6, -4, -6, -6, -0.9, 6.8, -1.6]).reshape(-1,1)
                z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2, 3.6]).reshape(-1,1)
            case 2:
                factor = 1.0
                x = factor*np.array([-5.0, 5.0,  5.0, -5.0,-5.0]).reshape(-1,1)
                y = factor*np.array([ 0.0, 0.0,  0.0,  0.0, 0.0]).reshape(-1,1)
                z = factor*np.array([ 5.0, 5.0, 15.0, 15.0, 5.0]).reshape(-1,1)
            case 3:
                x = np.array([0.0  ,  5.0, 0.0 , -5.0,   0.0,  5.0,  0.0, -5.0 , 0.0]).reshape(-1,1)
                y = np.array([-10.0, -5.0, 0.0 ,  5.0,  10.0,  5.0,  0.0, -5.0, -10.0]).reshape(-1,1)
                z = np.array([10.0 , 20.0, 10.0,  20.0, 10.0, 20.0, 10.0, 20.0 , 10.]).reshape(-1,1)
            case 4:
                x = np.array([ 0.0, 10.0, 10.0,  0.0, -10.0, -10.0, 0.0]).reshape(-1,1)
                y = np.array([ 0.0,  5.0, -5.0,  0.0,   5.0,  -5.0, 0.0]).reshape(-1,1)
                z = np.array([1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0]).reshape(-1,1)
        x_gates = np.concatenate([x.T,y.T,z.T]).T
        self.x_gates = x_gates
        self.s_tag = np.linspace(0,x.shape[0]-1,self.x_gates.shape[0])
        self.NumOfGates = x.shape[0]




@dataclass
class SplineCenterline(RaceTrack):
    def __init__(self, Traj: int):
        super().__init__(Traj)

        self.center_spline = sp.interpolate.CubicSpline(self.s_tag, self.x_gates, bc_type = 'periodic')

        self.xi = spline_interpolant(self.s_tag, self.x_gates[:,0], extrapolate = 'linear',
            bc_type = 'periodic',
            fast = False)
        self.xj = spline_interpolant(self.s_tag, self.x_gates[:,1], extrapolate = 'linear',
            bc_type = 'periodic',
            fast = False)
        self.xk = spline_interpolant(self.s_tag, self.x_gates[:,2], extrapolate = 'linear',
            bc_type = 'periodic',
            fast = False)
    
        sym_s = ca.SX.sym('s')
        
        xc = ca.vertcat(self.xi(sym_s), self.xj(sym_s), self.xk(sym_s))
        self.f_xc    = ca.Function('f_xc',    [sym_s], [xc])
        self.f_xcs   = ca.Function('f_xcs',   [sym_s], [ca.jacobian(self.f_xc(sym_s), sym_s)])
        self.f_xcss  = ca.Function('f_xcss',  [sym_s], [ca.jacobian(self.f_xcs(sym_s), sym_s)])
        self.f_xcsss = ca.Function('f_xcsss', [sym_s], [ca.jacobian(self.f_xcss(sym_s), sym_s)])

        # Calculate length between gates:
        self.sL = np.zeros(self.x_gates.shape[0])
        N_ds = 2000
        ds = 1.0/N_ds
        self.L = 0
        for i in range(self.x_gates.shape[0]-1):
            self.sL[i+1] += self.L
            for d in np.linspace(0,1,N_ds):
                deltaS = np.linalg.norm(self.f_xcs(self.sL[i]+d))*ds
                self.sL[i+1] += deltaS
                self.L += deltaS
        
        self.f_T = ca.Function('f_T', [sym_s], [self.f_xcs(sym_s) / ca.norm_2(self.f_xcs(sym_s))])
        self.f_B = ca.Function('f_B', [sym_s], [ca.cross(self.f_xcs(sym_s), self.f_xcss(sym_s)) / (ca.norm_2(ca.cross(self.f_xcs(sym_s), self.f_xcss(sym_s))))])
        self.f_N = ca.Function('f_N', [sym_s], [ca.cross(self.f_B(sym_s), self.f_T(sym_s))])
        self.f_k = ca.Function('f_k', [sym_s], [ca.norm_2(ca.cross(self.f_xcs(sym_s), self.f_xcss(sym_s)))/ca.fmax(0.001, ca.norm_2(self.f_xcs(sym_s)))**3])
        self.f_tau = ca.Function('f_tau', [sym_s], [ca.dot(ca.cross(self.f_xcs(sym_s),self.f_xcss(sym_s)),self.f_xcsss(sym_s))/ca.fmax(0.001, ca.norm_2(ca.cross(self.f_xcs(sym_s), self.f_xcss(sym_s))))**2])
        self.f_gate_factor = spline_interpolant([0.0, 0.5, 1.0], [1.0, 3.0, 1.0], extrapolate = 'linear', bc_type = 'clamped', fast = False)

        return


    def global_to_local(self, x: np.array, s_prev: float = None):
        
        s = self.s_prev if s_prev is None else s_prev
        if s > self.NumOfGates-1:
            s -= self.NumOfGates-1
        elif s < 0:
            s += self.NumOfGates-1
        x_s = self.f_xc(s)
        ds_sign = 1.0
        ds_step = 0.01
        n = np.dot((x-x_s).T, self.f_N(s))
        b = np.dot((x-x_s).T, self.f_B(s))
        delta = x - x_s - n*self.f_N(s) - b*self.f_B(s)
        while np.linalg.norm(delta) > 1e-4 and ds_step > 1e-4:
            s_dir = self.f_T(s)
            ds = np.dot((x-x_s).T, s_dir)
            if ds*ds_sign < 0:
                ds_sign = -ds_sign
                ds_step *= 0.5
            s += ds_sign*ds_step

            n = np.dot((x-x_s).T, self.f_N(s))
            b = np.dot((x-x_s).T, self.f_B(s))
            x_s = self.f_xc(s)
            delta = x - x_s - n*self.f_N(s) - b*self.f_B(s)

            # if s > self.NumOfGates-1:
            #     s -= self.NumOfGates-1
            # if s < 0:
            #     s += self.NumOfGates-1
        if abs(s-self.NumOfGates+1) < 1e-4:
            s = 0
        self.s_prev = s
        return s, n[0,0], b[0,0]

    def local_to_global(self, s: float, n: float, b: float):
            
        return self.f_xc(s) + n*self.f_N(s) + b*self.f_B(s)

    def get_gate_orientation(self, idx):
        return self.f_T(idx), self.f_N(idx), self.f_B(idx)

    def get_gate_transform(self, idx):
        return ca.vertcat(
            self.f_xc(idx),
            self.f_T(idx),
            self.f_N(idx),
            self.f_B(idx)
        )

    def Plot3D_Track(self, figNum: int = 100):
        
        s_vec = np.linspace(0,self.s_tag[-1],1000)
    # Create a spline for the centerline
        center_spline = np.zeros((s_vec.shape[0],3))
        for i in range(s_vec.shape[0]):
            center_spline[i] = np.array(self.f_xc(s_vec[i])).flatten()
    
        T = np.zeros((self.x_gates.shape[0],3))
        N = np.zeros((self.x_gates.shape[0],3))
        B = np.zeros((self.x_gates.shape[0],3))
        for i in range(self.x_gates.shape[0]):
            T[i,:] = np.array(self.f_T(i)).flatten()
            N[i,:] = np.array(self.f_N(i)).flatten()
            B[i,:] = np.array(self.f_B(i)).flatten()
        
        fig, ax = plt.subplots(3,2, figsize=(10,10))
        ax[0,0].remove()
        ax[1,0].remove()

        ax1 = fig.add_subplot(3,2,(1,3), projection='3d')
        ax1.plot3D(self.x_gates[:,0], self.x_gates[:,1],self.x_gates[:,2], 'ro', markersize=6)
        ax1.plot3D(center_spline[:,0], center_spline[:,1], center_spline[:,2], 'g--')

        ax1.grid('on')
        ax1.set_title('Track')
        if self.gate_type == 'square':
            gate = self.gate_size * np.array([[0,1,1],[0,1,-1],[0,-1,-1],[0,-1,1],[0,1,1]]).T
        elif self.gate_type == 'circle':
            gate = self.gate_size * np.array([np.zeros(100), np.cos(np.linspace(0,2*np.pi,100)), np.sin(np.linspace(0,2*np.pi,100))])

        for i in range(self.x_gates.shape[0]):
            theta = np.arctan2(T[i,1], T[i,0])
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
            gate_i = R @ gate
            ax1.plot3D(self.x_gates[i,0]+gate_i[0,:], self.x_gates[i,1]+gate_i[1,:], self.x_gates[i,2]+gate_i[2,:], 'm-')
        ax1.axis('equal')

        return fig, ax, ax1