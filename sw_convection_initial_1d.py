#!/usr/bin/env python
# coding: utf-8
from numba import jit, vectorize, float64
import numpy as np
import itertools
import h5py
import matplotlib.pyplot as plt
import dedalus.public as de
from dedalus.extras import flow_tools
import time
import logging
from convectiveParametrization import computecentersandtimes, heat_1d_serial
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)
np.seterr(all='raise')

#Domain specs
Lx                   = (1.0e6)
nx                   = (600)
# Create bases and domain
x_basis              = de.Fourier('x', nx, interval =(0, Lx), dealias=3./2)
domain               = de.Domain([x_basis], grid_dtype=np.float64)
local_shape = domain.dist.grid_layout.local_shape(scales=1.5)
conv_centers = np.zeros(local_shape, dtype=bool)
conv_centers_times = np.zeros(local_shape, dtype=np.float64)

#Set the parameters of the problem
##Numerics
#diff_coef            = 8.0e4 #Works with nu**2 lapl(lapl
diff_coef             = 1e11
#diff_coef            = 1.0e1 #Works with nu**2 lapl(lapl
#diff_coef = 1.0e3   #works with nu*nu lapl lapl
## Physics
gravity              = 10.0
#gravity = 0.0
coriolis_parameter   = 5e-4;
### Convective Params
heating_amplitude    = 1.0e10 #originally 9 for heating, -8 for cooling
radiative_cooling    = (1.12/3.0)*4*1.0e-3
#radiative_cooling    = (1.12/3)*1e-4
convective_timescale = 3600.0
convective_radius    = 6000.0
critical_geopotential = 40.0
damping_timescale = 0.5*86400.0
relaxation_height = 39.0

exp_name = 'f'+ format(coriolis_parameter,"1.0e")+'_q'+format(heating_amplitude,"1.0e")+'_r'+str(int(convective_radius/1000))+'_hc'+str(int(relaxation_height))
#exp_name='1d_tests'
output_path = '/Users/arreyes/Documents/Research/DedalusExperiments/DedalusOutput/'
#output_path = '/global/scratch/argelreyes/'

k                    = 2*np.pi/1000 #is wavelength = 1km
#k                    = 2.0*np.pi/10.0
H                    = 44.0#/(gravity*k**2)
omega                = np.sqrt(gravity*H*k**2) #wavelength to be resolved
# # *****USER-CFL***** #
dt_max               = (2*np.pi/omega)
CFLfac               = 0.5
start_dt             = dt_max

buf = bytearray(1<<21)

#**********Convective heating term*****************#

def ConvHeating(*args):
    t                    = args[0].value # this is a scalar; we use .value to get its value
    x                    = args[1].data # this is an array; we use .data to get its values
    h                    = args[2].data
    q0                   = args[3].value
    tauc                 = args[4].value
    R                    = args[5].value
    hc                   = args[6].value
    Lx                   = args[7].value
    Q                    = np.zeros_like(h)
    R2                   = R*R
#    if np.isnan(h[1,1]):
#        sys.exit("I have NaNs on the height field")
    xmax_local = np.amax(x)
    xmin_local = np.amin(x)
    computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times)
#    print(conv_centers)
    #### MPI version #######
    indices_out         = np.nonzero(conv_centers)
    centers_x     = np.array(x[indices_out])
    centers_times = np.array(conv_centers_times[indices_out])
    centers_shape       = centers_x.shape[0]
    numberofcenters     = np.sum(centers_shape)
    #print("Number of centers:")
    #print(numberofcenters)
   ### Q = 0
    if numberofcenters > 0:
        heat_1d_serial(Q,x,t,centers_x,centers_times,q0,tauc,R2,R,xmin_local,xmax_local,Lx)
    return Q



def DiabaticTerm(*args, domain=domain, F=ConvHeating):
    return de.operators.GeneralFunction(domain,layout='g',func=F,args=args)
#***** Problem setup ************#
de.operators.parseables['Q']        = DiabaticTerm

problem                             = de.IVP(domain, variables=['u','h','ux','hx'])
problem.substitutions['lapl(A)'] =  'dx(dx(A))'
problem.parameters['Lenx']            = Lx
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
#problem.add_equation("dt(u) + g*dx(h) - f*v +(nu**2)*lapl(lapl(u)) +u/taud = - u*ux ") #check it need changing for dx(h)
#problem.add_equation("dt(h)   +(nu**2)*lapl(lapl(h))   +h/taud  =- u*hx - h*ux + Q(t,x,h,q0,tauc,R,hc,Lenx) -r +h0/taud")
#### No coriolis
problem.add_equation("dt(u) + g*dx(h) +(nu)*lapl(lapl(u)) =  - u*ux  -u/taud") 
problem.add_equation("dt(h) + h/taud  +(nu)*lapl(lapl(h)) = - u*hx - h*ux + Q(t,x,h,q0,tauc,R,hc,Lenx) +h0/taud -r")

#1first order diffusion
#problem.add_equation("dt(u) + g*dx(h) -(nu**2)*(lapl(u)) =  - u*ux  -u/taud") 
#problem.add_equation("dt(h) + h/taud  -(nu**2)*(lapl(h)) = - u*hx - h*ux + Q(t,x,h,q0,tauc,R,hc,Lenx) +h0/taud -r")
#### No coriolis, no heating
#problem.add_equation("dt(u) = - g*hx - (nu**2)*dx(dx(u))   - u*ux  -u/taud") #check it need changing for dx(h)
#problem.add_equation("dt(h) = - (nu**2)*dx(dx(h))  - u*hx - h*ux  -(h-h0)/taud -r")
#problem.add_equation("dt(u)=1")
#problem.add_equation("dt(h)=1")
problem.add_equation("ux - dx(u) = 0")
problem.add_equation("hx - dx(h) = 0")



ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)

#******Field initialization********#


x = domain.grid(0)
u = solver.state['u']
ux = solver.state['ux']
h = solver.state['h']
hx = solver.state['hx']


amp    = 10
u['g'] = 0.0
np.random.seed(5)
#print(np.shape(h['g'].data))
nxlocal = h['g'].shape[0]
h['g'] = H - amp*np.random.rand(nxlocal)
#h['g'][int(nxlocal/2),int(nylocal/2)] = 30.0
#h['g'] = H - amp*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)
#h['g'] = H - amp*np.exp(- ((x-0.5*Lx)**2/(0.1*Lx)**2 ))

u.differentiate('x',out=ux)
h.differentiate('x',out=hx)




solver.stop_sim_time = 1*86400
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
initial_dt = dt_max
#print(initial_dt)
cfl = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=CFLfac,
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.5)
cfl.add_velocities(('u'))
cfl.add_nonconservative_diffusivity(('nu'))


analysis = solver.evaluator.add_file_handler(output_path+exp_name, sim_dt=3600, max_writes=300)
#analysis = solver.evaluator.add_file_handler(exp_name, iter=1, max_writes=300)
analysis.add_task('h',layout='g')
analysis.add_task('u',layout='g')
#analysis.add_task('u',layout='c')
analysis.add_task('x',layout='g')
solver.evaluator.vars['Lx'] = Lx


xplot = 1e-3*domain.grid(0,scales=domain.dealias)
xplot2 = [i for i,_ in enumerate(conv_centers)]
plt.ion()
fig, axis = plt.subplots(2,2,figsize=(10,5))
p1, = axis[0,0].plot(xplot,h['g'])
p2, = axis[0,1].plot(h['c'][1:]) #this gives a complex number warning
p3 = axis[1,0].bar(xplot2,conv_centers)
p4, = axis[1,1].plot(xplot,u['g'])

axis[1,0].set_yticks([0,1])
axis[1,0].set_yticklabels(['No','Yes'])
axis[1,0].set_xticks([])
axis[1,0].set_xticklabels([])
axis[0,0].set_xlabel('x (km)')
axis[0,0].set_ylabel('Height (m)')
axis[0,1].set_ylabel('Amplitude')
axis[0,1].set_xlabel('Wavenumber')
axis[1,1].set_xlabel('x (km)')
axis[1,1].set_ylabel('Velocity (m/s)')
#axis[0,0].set_ylim((0,100))
#axis[0,1].set_ylim(bottom=-5)
axis[1,0].set_ylim((0,1.1))
axis[0,0].set_title("Geopotential height")
axis[0,1].set_title("Geopotential height (spectrum)")
axis[1,1].set_title("Velocity")
axis[1,0].set_title("Is it convecting?")
fig.tight_layout()
axis[0,0].set_ylim((35,250))
plt.draw()
plt.pause(0.1)
#input("Press Enter to continue...")

logger.info('Starting loop')
start_run_time = time.time()
try:
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt,trim=False)
        if solver.iteration % 1000 == 0:        
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
        if (solver.iteration % 10 == 0) and (solver.sim_time % (15*86400) == 0): 
            #        input("Press Enter to continue...")
            fig.suptitle(f'Shallow Waters. Day: {solver.sim_time/86400:.1f}',y=0.98)
            p1.set_ydata(np.copy(h['g']))
            axis[0,0].hlines(y =  critical_geopotential,xmin=0,xmax=Lx, color ='r') 
            p2.set_ydata(h['c'][1:])
            p4.set_ydata(u['g'])
            [bar.set_height(conv_centers[i]) for i,bar in enumerate(p3)]
            #        axis[0,0].set_ylim(bottom=0)
            axis[0,0].relim()
            axis[0,0].autoscale_view()
            plt.draw()
            plt.pause(0.001)
            axis[0,1].relim()
            axis[0,1].autoscale_view()
            
            #axis[0,1].set_ylim(bottom=-5)
            axis[1,1].relim()
            axis[1,1].autoscale_view()
            plt.draw()
            plt.pause(0.0001)
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
    plt.show()



