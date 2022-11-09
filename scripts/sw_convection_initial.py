#!/usr/bin/env python
# coding: utf-8
import os
import sys
base_dir = os.path.realpath(os.path.dirname(__file__))+"/../"
sys.path.insert(0,base_dir+"src/")
print(base_dir)
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
from convectiveParametrization import computecentersandtimes, heat_mpi
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)
np.seterr(all='raise')

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

mpirankr = (mpirank - 1)%mpisize
mpirankl = (mpirank + 1)%mpisize

#Domain specs
Lx                   = (1.5e6)
nx                   = (250)
Ly                   = (1.0e6)
ny                   = (250)
# Create bases and domain
x_basis              = de.Fourier('x', nx, interval =(0, Lx), dealias=3./2)
y_basis              = de.Fourier('y', ny, interval=(0, Ly), dealias=3./2)
domain               = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

local_shape = domain.dist.grid_layout.local_shape(scales=1.5)
conv_centers = np.zeros(local_shape, dtype=bool)
conv_centers_times = np.zeros(local_shape, dtype=np.float64)

#Set the parameters of the problem
##Numerics
diff_coef            = 1.0e5 #I had 5
## Physics
gravity              = 9.8#gravity = 0.0
coriolis_parameter   = 5e-4;
### Convective Params
heating_amplitude    = 1.0e9 #originally 9 for heating, -8 for cooling
radiative_cooling    = (1.12/3.0)*1.0e-8
#radiative_cooling    = (1.12/3)*1e-4
convective_timescale = 28800.0
convective_radius    = 30000.0
critical_geopotential = 40.0
damping_timescale = 8.0*86400.0
relaxation_height = 39.0

exp_name = 'f'+ format(coriolis_parameter,"1.0e")+'_q'+format(heating_amplitude,"1.0e")+'_r'+str(int(convective_radius/1000))+'_hc'+str(int(relaxation_height))
#output_path = '/Users/arreyes/Documents/Research/DedalusExperiments/DedalusOutput/'
output_path = base_dir + '/data/'

k                    = 2*np.pi/1000 #is wavelength = 1km
#k                    = 2.0*np.pi/10.0
H                    = 39.0#/(gravity*k**2)
omega                = np.sqrt(gravity*H*k**2) #wavelength to be resolved

# # *****USER-CFL***** #
dt_max               = (2*np.pi/omega)# /19
CFLfac               = 0.5
start_dt             = dt_max

buf = bytearray(1<<21)

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
    Lx                   = args[8].value
    Ly                   = args[9].value
    Q                    = np.zeros_like(h)
    R2                   = R*R
#    if np.isnan(h[1,1]):
#        sys.exit("I have NaNs on the height field")
    xmax_local = np.amax(x)
    xmin_local = np.amin(x)
    ymax_local = np.amax(y)
    ymin_local = np.amin(y)
    #print("Process "+str(mpirank)+": Counting centers")
    computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times)
    #print("Process "+str(mpirank)+": Finished counting centers")
    #print("Rank: ",rank,"I have futures in: ",np.nonzero(conv_centers_times > t)," time: ",t)
    #### MPI version #######
    indices_out         = np.nonzero(conv_centers)
    centers_local_x     = np.array(x[indices_out])
    centers_local_y     = np.array(y[indices_out])
    centers_local_times = np.array(conv_centers_times[indices_out])
    comm.barrier()
    centers_shape       = comm.allgather(centers_local_x.shape[0])
    numberofcenters     = np.sum(centers_shape)
    #print(numberofcenters)
    comm.barrier()

    if numberofcenters > 0:
        #print("Process "+str(mpirank)+": Starting sharing centers with my brother processes")
         ###For times
        reqsl = comm.isend(np.array([centers_local_x,centers_local_y,centers_local_times]),dest=mpirankl)
        reqr = comm.irecv(buf,source=mpirankr)
        clr = reqr.wait()
        reqsl.wait()
        comm.barrier()
        reqsr = comm.isend([centers_local_x,centers_local_y,centers_local_times],dest=mpirankr)
        reql = comm.irecv(buf,source=mpirankl)
        cll = reql.wait()
        reqsr.wait()
        comm.barrier()
        #print("Process "+str(mpirank)+": Finished sharing centers with my brother processes")
        centers_gathered_x = np.hstack([cll[0],centers_local_x,clr[0]])
        centers_gathered_y = np.hstack([cll[1],centers_local_y,clr[1]])
        centers_gathered_times = np.hstack([cll[2],centers_local_times,clr[2]])

        
        # print("")
         
     #   centers_gathered_y     = np.hstack(comm.allgather(centers_local_y)[[mpirankl,mpirank,mpirankr]] )
     #   centers_gathered_times = np.hstack(comm.allgather(centers_local_times)[[mpirankl,mpirank,mpirankr]] )
         
        
        #print("My rank is: ",mpirank," My xs are: ",centers_gathered_x)

        #print("My rank is: ",mpirank," My ys are: ",centers_gathered_y)
        #print("My rank is: ",mpirank," My ts are: ",centers_gathered_times)
         #print("Rank: ",rank,"I have futures in: ",np.nonzero(centers_gathered_times > t))
        #print("Process "+str(mpirank)+": Starting heating")
        heat_mpi(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local,Lx,Ly)
        #print("Process "+str(mpirank)+": Finished heating")
    ##### Serial version #######
    #heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2)
    #print("Time",t,"max onvective heating: ",np.amax(Q))
    return Q



def DiabaticTerm(*args, domain=domain, F=ConvHeating):
    return de.operators.GeneralFunction(domain,layout='g',func=F,args=args)
#***** Problem setup ************#
de.operators.parseables['Q']        = DiabaticTerm

problem                             = de.IVP(domain, variables=['u','v','h','ux','hx','vy','hy','uy','vx'])
#problem.meta['u']['x']['dirichlet'] = True
problem.substitutions['diff(A)'] =  '(-1)**(hs+1)*nu*dx(dx(A)) + (-1)**(hs+1)*nu*dy(dy(A))'
problem.substitutions['lapl(A)'] =  'dx(dx(A)) + dy(dy(A))'

#problem.substitutions['damp(A,A0,tau)'] = '(A-A0)/tau'
problem.parameters['Lenx']            = Lx
problem.parameters['Leny']            = Ly
problem.parameters['g']             = gravity
problem.parameters['nu']            = diff_coef
problem.parameters['f']             = coriolis_parameter
problem.parameters['q0']            = heating_amplitude
problem.parameters['tauc']          = convective_timescale
problem.parameters['R']             = convective_radius
problem.parameters['hc']            = critical_geopotential
problem.parameters['r']             = radiative_cooling
problem.parameters['taud']          = damping_timescale
problem.parameters['h0']            = relaxation_height
### Full physics
#problem.add_equation("dt(u) + g*dx(h) - f*v -(nu)*lapl(u) +u/taud = - u*ux - v*uy ") #check it need changing for dx(h)
#problem.add_equation("dt(v) + g*dy(h) + f*u   -(nu)*lapl(v) +v/taud = - v*vy - u*vx") #check it need changing for dx(h)
#problem.add_equation("dt(h)   -(nu)*lapl(h)   +h/taud  =- u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,q0,tauc,R,hc,Lenx,Leny) -r +h0/taud")
#### No coriolis
problem.add_equation("dt(u) + g*dx(h) - f*v -(nu)*lapl(u) +u/taud = - u*ux - v*uy ") #check it need changing for dx(h)
problem.add_equation("dt(v) + g*dy(h) + f*u   -(nu)*lapl(v) +v/taud = - v*vy - u*vx") #check it need changing for dx(h)
problem.add_equation("dt(h)   -(nu)*lapl(h)   +h/taud  =- u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,q0,tauc,R,hc,Lenx,Leny) -r +h0/taud")

#### No coriolis, no heating
#problem.add_equation("dt(u) + g*dx(h)  = (diff(u))  - u*ux - v*uy  -u/taud") #check it need changing for dx(h)
#problem.add_equation("dt(v) + g*dy(h)  = (diff(v))  - v*vy - u*vx -v/taud") #check it need changing for dx(h)
#problem.add_equation("dt(h)  = (diff(h))   - u*hx - h*ux - h*vy - v*hy  -(h-h0)/taud -r")


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

amp    = 10.0
np.random.seed(mpirank)
u['g'] = 0.0
v['g'] = 0.0
#h['g'] = H - amp*np.random.rand(nxlocal,nylocal)
#h['g'] = H - amp*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)
#h['g'] = H - amp*np.exp(- ((x-0.5*Lx)**2/(0.1*Lx)**2 + (y-0.5*Ly)**2/(0.1*Ly)**2 ))
np.random.seed(5)
phase  =  Lx*np.random.rand()
wavenum = np.random.randint(1, 10)
#print(np.shape(h['g'].data))
nxlocal = h['g'].shape[0]
nylocal = h['g'].shape[1]
h['g'] = H + amp*np.random.rand(nxlocal, nylocal)
#h['g'][int(nxlocal/2),int(nylocal/2)] = 30.0

#h['g'] = H - amp*np.sin(wavenum*np.pi*x/Lx + phase)*np.sin(wavenum*np.pi*y/Ly+phase)+np.random.rand(nxlocal,nylocal)
#h['g'] = H - amp*np.exp(- ((x-0.5*Lx)**2/(0.1*Lx)**2 ))

u.differentiate('x',out=ux)
v.differentiate('y',out=vy)
v.differentiate('x',out=vx)
u.differentiate('y',out=uy)
h.differentiate('x',out=hx)
h.differentiate('y',out=hy)



solver.stop_sim_time = 20*86400
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
initial_dt = dt_max
#print(initial_dt)
cfl = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=CFLfac,
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.5)
#initial_dt = 0.2*Lx/nx
#cfl = flow_tools.CFL(solver,initial_dt=initial_dt,cadence=10,safety=0.5,threshold=0.5)
cfl.add_velocities(('u','v'))



analysis = solver.evaluator.add_file_handler(output_path+exp_name, sim_dt=3600, max_writes=300)
#analysis = solver.evaluator.add_file_handler(exp_name, iter=1, max_writes=300)
analysis.add_task('h',layout='g')
analysis.add_task('u',layout='g')
analysis.add_task('v',layout='g')
analysis.add_task('y',layout='g')
analysis.add_task('x',layout='g')
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly

logger.info('Starting loop')
start_run_time = time.time()
try:
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt,trim=False)
        if solver.iteration % 1 == 0:
            logger.info('Iteration: %i, Time: %.1f, dt: %e' %(solver.iteration, solver.sim_time/86400, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Output was stored in %s' %(output_path+exp_name))




