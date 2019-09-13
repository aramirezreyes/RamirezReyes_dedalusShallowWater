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
from convectiveParametrization import computecentersandtimes, heat_mpi2
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)
np.seterr(all='raise')
#import matplotlib.pyplot as plt
#%matplotlib inline

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

mpirankr = (mpirank - 1)%mpisize
mpirankl = (mpirank + 1)%mpisize




#Domain specs
Lx                   = (1.0e6)
nx                   = (200)
Ly                   = (1.5e6)
ny                   = (200)
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
diff_coef            = 1.0e4 #I had 5
hyperdiff_power      = 1.0
## Physics
gravity              = 10.0
#gravity = 0.0
coriolis_parameter   = 5e-4;
### Convective Params
heating_amplitude    = 3.0e13 #originally 9 for heating, -8 for cooling
radiative_cooling    = (1.12/3.0)*1.0e-8
#radiative_cooling    = (1.12/3)*1e-4
convective_timescale = 28800.0
convective_radius    = 30000.0
critical_geopotential = 40.0
damping_timescale = 2.0*86400.0
relaxation_height = 39.0

k                    = 2*np.pi/2000 #1000 is wavelength = 1km
#k                    = 2.0*np.pi/10.0
H                    = 40.0#/(gravity*k**2)
omega                = np.sqrt(gravity*H*k**2) #wavelength to be resolved
# #Initialize fields
# #conv_centers['g']         = 0.0
# #conv_centers_times['g']   = 0.0


# # *****USER-CFL***** #
dt_max               = (2*np.pi/omega) #/19
CFLfac               = 0.8
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
    #print("Heating")
    computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times)
    #print("Rank: ",rank,"I have futures in: ",np.nonzero(conv_centers_times > t)," time: ",t)
    #### MPI version #######
    indices_out         = np.nonzero(conv_centers)
    centers_local_x     = x[indices_out]
    centers_local_y     = y[indices_out]
    centers_local_times = conv_centers_times[indices_out]
    comm.barrier()
    centers_shape       = comm.allgather(centers_local_x.shape[0])
    numberofcenters     = np.sum(centers_shape)
    #print(numberofcenters)
    comm.barrier()
    
    if numberofcenters > 0:
         centers_gathered_x = np.array(comm.allgather(centers_local_x))[[mpirankl,mpirank,mpirankr]]
         centers_gathered_x = np.hstack(centers_gathered_x)

         centers_gathered_y = np.array(comm.allgather(centers_local_y))[[mpirankl,mpirank,mpirankr]]
         centers_gathered_y  = np.hstack(centers_gathered_y)

         centers_gathered_times = np.array(comm.allgather(centers_local_times))[[mpirankl,mpirank,mpirankr]]
         centers_gathered_times = np.hstack(centers_gathered_times)
         
         #centers_gathered_y     = np.hstack(comm.allgather(centers_local_y)[[mpirankl,mpirank,mpirankr]] )
         #centers_gathered_times = np.hstack(comm.allgather(centers_local_times)[[mpirankl,mpirank,mpirankr]] )
         comm.barrier()
         #print("Rank: ",rank,"I have futures in: ",np.nonzero(centers_gathered_times > t))
         heat_mpi2(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local,Lx,Ly)
    ##### Serial version #######
    #heat(Q,x,y,t,conv_centers,conv_centers_times,q0,tauc,R2)
    return Q



def DiabaticTerm(*args, domain=domain, F=ConvHeating):
    return de.operators.GeneralFunction(domain,layout='g',func=F,args=args)
#***** Problem setup ************#
de.operators.parseables['Q']        = DiabaticTerm

problem                             = de.IVP(domain, variables=['u','v','h','ux','hx','vy','hy','uy','vx'])
#problem.meta['u']['x']['dirichlet'] = True
problem.substitutions['diff(A)'] =  '(-1)**(hs+1)*nu*dx(dx(A)) + (-1)**(hs+1)*nu*dy(dy(A))'
#problem.substitutions['damp(A,A0,tau)'] = '(A-A0)/tau'
problem.parameters['Lenx']            = Lx
problem.parameters['Leny']            = Ly
problem.parameters['g']             = gravity
problem.parameters['nu']            = diff_coef
problem.parameters['hs']            = hyperdiff_power
problem.parameters['f']             = coriolis_parameter
problem.parameters['q0']            = heating_amplitude
problem.parameters['tauc']          = convective_timescale
problem.parameters['R']             = convective_radius
problem.parameters['hc']            = critical_geopotential
problem.parameters['r']             = radiative_cooling
problem.parameters['taud']          = damping_timescale
problem.parameters['h0']            = relaxation_height
problem.add_equation("dt(u) + g*dx(h) - f*v  = (diff(u))  - u*ux - v*uy  -u/taud") #check it need changing for dx(h)
problem.add_equation("dt(v) + g*dy(h) + f*u  = (diff(v))  - v*vy - u*vx -v/taud") #check it need changing for dx(h)
problem.add_equation("dt(h)  = (diff(h))   - u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,q0,tauc,R,hc,Lenx,Leny) -(h-h0)/taud -r")

#problem.add_equation("dt(u) + g*dx(h)  = (diff(u))  - u*ux - v*uy  -u/taud") #check it need changing for dx(h)
#problem.add_equation("dt(v) + g*dy(h)  = (diff(v))  - v*vy - u*vx -v/taud") #check it need changing for dx(h)
#problem.add_equation("dt(h)  = (diff(h))   - u*hx - h*ux - h*vy - v*hy + Q(t,x,y,h,q0,tauc,R,hc,Lenx,Leny) -(h-h0)/taud -r")


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

amp    = 4.0
u['g'] = 0.0
v['g'] = 0.0
#h['g'] = H - amp*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)
#h['g'] = H - amp*np.exp(- ((x-0.5*Lx)**2/(0.1*Lx)**2 + (y-0.5*Ly)**2/(0.1*Ly)**2 ))
#print(np.shape(h['g'].data))
nxlocal = h['g'].shape[0]
nylocal = h['g'].shape[1]


np.random.seed(mpirank)
h['g'] = H + amp*np.random.rand(nxlocal,nylocal)
#h['g'][int(nxlocal/2),int(nylocal/2)] = 30.0

u.differentiate('x',out=ux)
v.differentiate('y',out=vy)
v.differentiate('x',out=vx)
u.differentiate('y',out=uy)
h.differentiate('x',out=hx)
h.differentiate('y',out=hy)



solver.stop_sim_time = 100*86400
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
initial_dt = dt_max
#print(initial_dt)
cfl = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=CFLfac,
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.5)
#initial_dt = 0.2*Lx/nx
#cfl = flow_tools.CFL(solver,initial_dt=initial_dt,cadence=10,safety=0.5,threshold=0.5)
cfl.add_velocities(('u','v'))



analysis = solver.evaluator.add_file_handler('analysis_convgravicor_bigdt', sim_dt=900, max_writes=300)
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

