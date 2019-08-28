#!/usr/bin/env python
# coding: utf-8
from numba import jit
from numba import vectorize, float64
import numpy as np
import itertools
import h5py
import dedalus.public as de
from dedalus.extras import flow_tools
import time
from mpi4py import MPI
import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

#import matplotlib.pyplot as plt
#%matplotlib inline

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Domain specs
Lx                   = (1.5e6)
nx                   = (150)
Ly                   = (1.5e6)
ny                   = (150)
# Create bases and domain
x_basis              = de.Fourier('x', nx, interval =(0, Lx), dealias=3./2)
y_basis              = de.Fourier('y', ny, interval=(0, Ly), dealias=3./2)
domain               = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
#Pretty sure that what you want to do is get the local shape of the grid on each process at the beginning of the run:
local_shape = domain.dist.grid_layout.local_shape(scales=1.5)
#Then create NumPy arrays on each processor locally that do what you're interested in :
conv_centers = np.zeros(local_shape, dtype=bool)
conv_centers_times = np.zeros(local_shape, dtype=np.float64)

#Set the parameters of the problem
##Numerics
diff_coef            = 1.0e4
hyperdiff_power      = 1.0
## Physics
gravity              = 10.0
#gravity = 0.0
coriolis_parameter   = 5e-4;
### Convective Params
heating_amplitude    = 1.0e11 #originally 9 for heating, -8 for cooling
radiative_cooling    = (1.12/3.0)*1.0e-8
#radiative_cooling    = (1.12/3)*1e-4
convective_timescale = 28800.0
convective_radius    = 20000.0
critical_geopotential = 40.0
k                    = 2*np.pi/1000 #1000 is wavelength = 1km
#k                    = 2.0*np.pi/10.0
H                    = 40.0#/(gravity*k**2)
omega                   = np.sqrt(gravity*H*k**2) #wavelength to be resolved
# #Initialize fields
# #conv_centers['g']         = 0.0
# #conv_centers_times['g']   = 0.0


# # *****USER-CFL***** #
dt_max               = (2*np.pi/omega) #/19
CFLfac               = 0.4
start_dt             = dt_max
#**********Convective heating term*****************#
def ConvHeating(*args):
    t                    = args[0].value # this is a scalar; we use .value to get its value
    x                    = args[1].data # this is an array; we use .data to get its values
    y                    = args[2].data
    h                    = args[3].data
    q0                   = args[4].value
    tauc                 = args[5].value
    R                    = args[6].value
    hc                   = args[7].value
    Q                    = np.zeros_like(h)
    R2                   = R*R
    #print("Sizes of x, y, conv_centers and h: ",np.shape(x),np.shape(y),np.shape(conv_centers),np.shape(h))
    """
    for each active center:
    for each point closer than R:
    compute q
    if dt > tc
    active center = 0 
    active center time = 0
    
    """
#    print("Conv_centers",conv_centers.shape)
#    print("Grid:",x.shape)
    computecentersandtimes2(t,h,hc,tauc,conv_centers,conv_centers_times)
    #print("From the inside: My rank, min(x), max(x), min (y), max(y) ",rank, np.amin(x),np.amax(x),np.amin(y),np.amax(y))
    xmax_local = np.amax(x)
    xmin_local = np.amin(x)
    ymax_local = np.amax(y)
    ymin_local = np.amin(y)
    #### MPI version #######
    indices_out         = np.nonzero(conv_centers)
    centers_local_x     = x[indices_out]
    centers_local_y     = y[indices_out]
    centers_local_times = conv_centers_times[indices_out]
    comm.barrier()
    centers_shape       = comm.allgather(centers_local_x.shape[0])
    numberofcenters     = np.sum(centers_shape)
    #print(numberofcenters)
    if numberofcenters > 0:
        centers_gathered_x     = np.hstack(comm.allgather(centers_local_x))
        centers_gathered_y     = np.hstack(comm.allgather(centers_local_y))
        centers_gathered_times = np.hstack(comm.allgather(centers_local_times))
        heat_mpi2(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local)
    ##### Serial version #######
    #heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2)
    return Q


@jit(nopython=True,parallel=True)
def heat_mpi(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2):    
    for ind in range(centers_gathered_x.shape[0]):
        xx              = centers_gathered_x[ind]
        yy              = centers_gathered_y[ind]
        time_convecting = centers_gathered_times[ind]
        distsq          = (x-xx)**2 + (y-yy)**2
        indices_in      = np.nonzero(distsq < R2)
        if indices_in[0].shape[0] > 0:
            for ind_in in range(indices_in[0].shape[0]):
                idxx_in            = indices_in[0][ind_in]
                idxy_in            = indices_in[1][ind_in]
                Q[idxx_in,idxy_in] = Q[idxx_in,idxy_in] + heatingfunction(t,distsq[idxx_in,idxy_in],time_convecting,q0,tauc,R2)    
    return None


@jit(nopython=True)
def heat_mpi2(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local):    
#    for index_out,val_out in range(centers_gathered_x.shape[0]):
    for index_out,val_out in np.ndenumerate(centers_gathered_x):
        xx              = val_out
        yy              = centers_gathered_y[index_out]
        time_convecting = centers_gathered_times[index_out]
        if xx > (xmax_local + R) or xx < (xmin_local - R) or yy < (ymin_local - R) or yy > (ymax_local + R ):
            continue
        else:
            for index_in,val_in in np.ndenumerate(x):
                distsq = (val_in-xx)**2+(y[index_in]-yy)**2
                if distsq < R2:
                    Q[index_in] = Q[index_in] + heatingfunction(t,distsq,time_convecting,q0,tauc,R2)    
    return None




@jit(nopython=True)
def heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2):
    indices_out = np.nonzero( (conv_centers!=0.0) & ((t - conv_centers_times)<tauc))
    for ind in range(indices_out[1].shape[0]):
        idxx = indices_out[0][ind]
        idxy = indices_out[1][ind]        
        distsq = (x-x[idxx,idxy])**2 + (y-y[idxx,idxy])**2        
        indices_in = np.nonzero(distsq < R2)
        for ind_in in range(indices_in[1].shape[0]):
            idxx_in = indices_in[0][ind_in]
            idxy_in = indices_in[1][ind_in]
            Q[idxx_in,idxy_in] = Q[idxx_in,idxy_in] +  heatingfunction(t,distsq[idxx_in,idxy_in],conv_centers_times[idxx,idxy],q0,tauc,R2)    
    return None


@jit(nopython=True,parallel=True)
def computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times):
    """ This functions takes arrays """
    ind1                       = np.nonzero((t - conv_centers_times)  >= tauc)
    for ind in range(ind1[1].shape[0]):
        conv_centers[ind1[0][ind],ind1[1][ind]] = False
        conv_centers_times[ind1[0][ind],ind1[1][ind]] = 0.0
        
    ind2                       = np.nonzero(h >= hc)
    for ind in range(ind2[1].shape[0]):
        conv_centers[ind2[0][ind],ind2[1][ind]] = False
        conv_centers_times[ind2[0][ind],ind2[1][ind]] = 0.0
        
    ind3                       = np.nonzero(h<hc)
    ind4                       = np.nonzero((h<hc) & (conv_centers_times == 0.0))
    
    for ind in range(ind3[1].shape[0]):
        conv_centers[ ind3[0][ind], ind3[1][ind]] = True
    for ind in range(ind4[1].shape[0]):
        conv_centers_times[ind4[0][ind],ind4[1][ind]] = t
    return None

@jit(nopython=True)
def computecentersandtimes2(t,h,hc,tauc,conv_centers,conv_centers_times):
    """ This functions takes arrays """
    for ind,val in np.ndenumerate(h):
        if val > hc:
            conv_centers[ind] = False
            conv_centers_times[ind] = 0.0
        else:
            conv_centers[ind] = True
            if conv_centers_times[ind] == 0.0 or ((t - conv_centers_times[ind]) >= tauc):
                conv_centers_times[ind] = t
            else:
                continue        
    return None
#@vectorize([float64(float64,float64,float64,float64,float64,float64)])
@jit(nopython=True)
def heatingfunction(t,distsq,conv_center_time,q0,tauc,R2):
    """ This function must receive strictly floats """                       
    A0       = np.pi*R2
    deltat   = t - conv_center_time
    quotient = 2.0 * (deltat - tauc/2.0)/(tauc)
    q        = q0*(1.0 - quotient*quotient)*(1.0 - (distsq / R2))

    return  q / (tauc*A0)



def DiabaticTerm(*args, domain=domain, F=ConvHeating):
    return de.operators.GeneralFunction(domain,layout='g',func=F,args=args)
#***** Problem setup ************#
de.operators.parseables['Q']        = DiabaticTerm

problem                             = de.IVP(domain, variables=['u','v','h','ux','hx','vy','hy','uy','vx'])
#problem.meta['u']['x']['dirichlet'] = True
#problem.parameters['conv_centers']  = conv_centers
#problem.parameters['conv_centers_times'] = conv_centers_times


problem.parameters['g']             = gravity
problem.parameters['nu']            = diff_coef
problem.parameters['hs']            = hyperdiff_power
problem.parameters['f']             = coriolis_parameter
problem.parameters['q0']            = heating_amplitude
problem.parameters['tauc']          = convective_timescale
problem.parameters['R']             = convective_radius
problem.parameters['hc']            = critical_geopotential
problem.parameters['r']             = radiative_cooling
problem.add_equation("dt(u) + g*dx(h) - f*v  - (-1)**(hs+1)*nu*dx(dx(u)) - (-1)**(hs+1)*nu*dy(dy(u)) = - u*ux - v*uy ") #check it need changing for dx(h)
problem.add_equation("dt(v) + g*dy(h) + f*u  - (-1)**(hs+1)*nu*dx(dx(v)) - (-1)**(hs+1)*nu*dy(dy(v)) = - v*vy - u*vx ") #check it need changing for dx(h)
problem.add_equation("dt(h)  - (-1)**(hs+1)*nu*dx(dx(h)) - (-1)**(hs+1)*nu*dy(dy(h)) = - u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,q0,tauc,R,hc) - r ")
problem.add_equation("ux - dx(u) = 0")
problem.add_equation("hx - dx(h) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("hy - dy(h) = 0")
problem.add_equation("vx - dx(v) = 0")
problem.add_equation("uy - dy(u) = 0")

ts = de.timesteppers.RK443


solver =  problem.build_solver(ts)

#******Field initialization********#

x = domain.grid(0)
y = domain.grid(1)
#xmax_local = np.amax(x)
#xmin_local = np.amin(x)
#ymax_local = np.amax(y)
#ymin_local = np.amin(y)
u = solver.state['u']
ux = solver.state['ux']
uy = solver.state['uy']
v = solver.state['v']
vx = solver.state['vx']
vy = solver.state['vy']
h = solver.state['h']
hx = solver.state['hx']
hy = solver.state['hy']


#print("From the outside: My rank, min(x), max(x), min (y), max(y) ",rank, np.amin(x),np.amax(x),np.amin(y),np.amax(y))

amp    = 4.0
u['g'] = 0.0
v['g'] = 0.0
#h['g'] = H - amp*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)
#h['g'] = H - amp*np.exp(- ((x-0.5*Lx)**2/(0.1*Lx)**2 + (y-0.5*Ly)**2/(0.1*Ly)**2 ))
#print(np.shape(h['g'].data))
nxlocal = h['g'].shape[0]
nylocal = h['g'].shape[1]



h['g'] = H - amp*np.random.rand(nxlocal,nylocal)
#h['g'][int(nxlocal/2),int(nylocal/2)] = 30.0

u.differentiate('x',out=ux)
v.differentiate('y',out=vy)
v.differentiate('x',out=vx)
u.differentiate('y',out=uy)
h.differentiate('x',out=hx)
h.differentiate('y',out=hy)



solver.stop_sim_time = 864000
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
initial_dt = dt_max
#print(initial_dt)
cfl = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=CFLfac,
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.5)
#initial_dt = 0.2*Lx/nx
#cfl = flow_tools.CFL(solver,initial_dt=initial_dt,cadence=10,safety=0.5,threshold=0.5)
cfl.add_velocities(('u','v'))



analysis = solver.evaluator.add_file_handler('analysis_convgravicor_bigdt', sim_dt=7200, max_writes=300)
#analysis = solver.evaluator.add_file_handler('analysis_convgravicor_bigdt', iter=1, max_writes=300)
analysis.add_task('h',layout='g')
analysis.add_task('u',layout='g')
analysis.add_task('v',layout='g')
analysis.add_task('y',layout='g')
analysis.add_task('x',layout='g')
#analysis.add_task('Q',layout='g')
#analysis.add_task('conv_centers',layout='g')
#analysis.add_task('conv_centers_times',layout='g')
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly



logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt,trim=False)
    if solver.iteration % 100 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

