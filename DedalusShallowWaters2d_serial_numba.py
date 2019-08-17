#!/usr/bin/env python
# coding: utf-8
from numba import jit
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


#Domain specs
Lx                   = (1.5e6)
nx                   = (150)
Ly                   = (1.0e6)
ny                   = (150)
# Create bases and domain
x_basis              = de.Fourier('x', nx, interval =(0, Lx), dealias=3./2)
y_basis              = de.Fourier('y', ny, interval=(0, Ly), dealias=3./2)
domain               = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
conv_centers         = domain.new_field(name = 'conv_centers')
conv_centers_times   = domain.new_field(name = 'conv_centers_times')
#Set the parameters of the problem
##Numerics
diff_coef            = 1.0e4
hyperdiff_power      = 1
## Physics
gravity              = 10.0
#gravity = 0.0
coriolis_parameter   = 5e-4;
### Convective Params
heating_amplitude    = 1.0e9
radiative_cooling    = (1.12/3.0)*1e-8

convective_timescale = 28800.0
convective_radius    = 20000.0
critical_geopotential = 40.0
#k                    = 2*np.pi/1000
k                    = 2.0*np.pi/10.0
H                    = 40.0#/(gravity*k**2)
om                   = np.sqrt(gravity*H*k**2)
#Initialize fields
conv_centers['g']         = 0.0
conv_centers_times['g']   = 0.0


# *****USER-CFL***** #
dt_max               = (2*np.pi/om) #/19
CFLfac               = 0.4
start_dt             = dt_max
#**********Convective heating term*****************#

def ConvHeating(*args):
    t                    = args[0].value # this is a scalar; we use .value to get its value
    x                    = args[1].data # this is an array; we use .data to get its values
    y                    = args[2].data
    h                    = args[3].data
    conv_centers         = args[4].data
    conv_centers_times   = args[5].data
    q0                   = args[6].value
    tauc                 = args[7].value
    R                    = args[8].value
    hc                   = args[9].value
    Q = np.zeros_like(h)
    R2                   = R*R
    ind_dist_x             = int(np.ceil( R / (x[1,0] - x[0,0]) ) )
    ind_dist_y             = int(np.ceil( R / (y[0,1] - y[0,0]) ) )
    #print(np.shape(x))
    #print()
    """
    for each active center:
    for each point closer than R:
    compute q
    if dt > tc
    active center = 0 
    active center time = 0
    
    """
    #print(np.shape(h))
    #print("Computed heating.")
    computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times)
    heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2,ind_dist_x,ind_dist_y)
    

    return Q

@jit(nopython=True)
def heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2,ind_dist_x,ind_dist_y):
    indices_out = np.nonzero( (conv_centers!=0.0) & ((t - conv_centers_times)<tauc))
    #print(indices_out)
    for ind in range(indices_out[1].shape[0]):
        idxx = indices_out[0][ind]
        idxy = indices_out[1][ind]        
        distsq = (x-x[idxx,idxy])**2 + (y-y[idxx,idxy])**2        
        indices_in = np.nonzero(distsq < R2)
        for ind_in in range(indices_in[1].shape[0]):
            idxx_in = indices_in[0][ind_in]
            idxy_in = indices_in[1][ind_in]
            #print("Shape dist: ",np.shape(distsq[mask])," Shape maks: ",np.shape(mask),"Shape Q",np.shape(hetin))
            Q[idxx_in,idxy_in] = Q[idxx_in,idxy_in] +  heatingfunction(t,distsq[idxx_in,idxy_in],conv_centers_times[idxx,idxy],q0,tauc,R2)    
    return None


@jit(nopython=True)
def computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times):
    """ This functions takes arrays """
    ind1                       = np.nonzero(h<hc)
    ind2                       = np.nonzero( (h<hc) & (conv_centers_times==0.0))
    ind3                       = np.nonzero(h > hc)
    ind4                       = np.nonzero(conv_centers_times > tauc)
    for ind in range(ind1[1].shape[0]):
        conv_centers[ ind1[0][ind], ind1[1][ind]] = 1.0
                     
    for ind in range(ind2[1].shape[0]):
        conv_centers_times[ind2[0][ind],ind2[1][ind]] = t
                     
    for ind in range(ind3[1].shape[0]):
        conv_centers[ind3[0][ind],ind3[1][ind]] = 0.0
        conv_centers_times[ind3[0][ind],ind3[1][ind]] = 0.0             
                           
    for ind in range(ind4[1].shape[0]):
        conv_centers_times[ind4[0][ind],ind4[1][ind]] = 0.0
                     
    
#    conv_centers[ ind1  ]      = 1.0
#    conv_centers_times[ ind2 ] = t
#    conv_centers[ind3]         = 0.0
#    conv_centers_times[ind3]   = 0.0
#    conv_centers_times[ind4]   = 0.0            
    #print(np.column_stack((x[idx],y[idx])))
    return None

@jit(nopython=True)
def heatingfunction(t,distsq,conv_center_time,q0,tauc,R2):
    """ This function must receive strictly floats """                       
    A0       = np.pi*R2
    deltat   = t - conv_center_time
    quotient = 2.0 * (deltat - tauc/2.0)/(tauc)
    q        = q0*(1.0 - quotient*quotient)*(1.0 - (distsq / R2))
    #print("Shape deltat: ",np.shape(conv_center_time)," Shape quotient: ",np.shape(quotient),"Shape q",np.shape(q))
    #print(np.shape(q))
    #return 1.0
    return  q / (tauc*A0)

def DiabaticTerm(*args, domain=domain, F=ConvHeating):
    return de.operators.GeneralFunction(domain,layout='g',func=F,args=args)



#######List of centers and times



#***** Problem setup ************#
de.operators.parseables['Q']        = DiabaticTerm
problem                             = de.IVP(domain, variables=['u','v','h','ux','hx','vy','hy','uy','vx'])
problem.meta['u']['x']['dirichlet'] = True
problem.parameters['conv_centers']  = conv_centers
problem.parameters['conv_centers_times'] = conv_centers_times
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
problem.add_equation("dt(h) - (-1)**(hs+1)*nu*dx(dx(h)) - (-1)**(hs+1)*nu*dy(dy(h)) = - u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,conv_centers,conv_centers_times,q0,tauc,R,hc) - r ")
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
u = solver.state['u']
ux = solver.state['ux']
uy = solver.state['uy']
v = solver.state['v']
vx = solver.state['vx']
vy = solver.state['vy']
h = solver.state['h']
hx = solver.state['hx']
hy = solver.state['hy']


amp    = 0.1
u['g'] = 0.0
v['g'] = 0.0
#h['g'] = H - amp*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)
#h['g'] = H - amp*np.exp(-4*np.log(2)*( (x-0.5*Lx)**2 + (y-0.5*Ly)**2)/(0.1*Lx)**2)
#print(np.shape(h['g'].data))
h['g'] = H + amp*np.random.rand(nx,ny)

u.differentiate('x',out=ux)
v.differentiate('y',out=vy)
v.differentiate('x',out=vx)
u.differentiate('y',out=uy)
h.differentiate('x',out=hx)
h.differentiate('y',out=hy)



solver.stop_sim_time = 8640000 #30000
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
initial_dt = 100
#print(initial_dt)
cfl = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=CFLfac,
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.5)
#initial_dt = 0.2*Lx/nx
#cfl = flow_tools.CFL(solver,initial_dt,safety=0.5)
cfl.add_velocities(('u','v'))



analysis = solver.evaluator.add_file_handler('analysis_convgravicor_bigdt', sim_dt=3600, max_writes=300)
#analysis = solver.evaluator.add_file_handler('analysis_convgravicor', iter=10, max_writes=300)
analysis.add_task('h',layout='g')
analysis.add_task('u',layout='g')
analysis.add_task('v',layout='g')
analysis.add_task('y',layout='g')
analysis.add_task('x',layout='g')
#analysis.add_task('Q',layout='g')
analysis.add_task('conv_centers',layout='g')
analysis.add_task('conv_centers_times',layout='g')
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly



logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt,trim=False)
    if solver.iteration % 20 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

