from numba import jit
import numpy as np

"""
heat_mpi(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local)

Receives the positions of the centers of convection and fills the matrix Q with the convective heating terms.

Q is an array of the same shape as h
x,y are arrays containing the coordinates of the portion of the domain that is local to this mpi process
t is an scalar with the current time
centers_gathered_x,centers_gathered_y and centers_gathered_times are arrays of the cordinates of the convecting points and the times at which they started convecting
q0, tauc, R2, R are floats with the parameters for the convection scheme
xmin_local, xmax_local, ymin_local, ymax_local are floats with the limits of the local portion of the domain

This version considers periodic boundaries by calling the ghosts routine with python lists (preferred)

"""

@jit(nopython=True,cache=False)
def heat_mpi(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local,Lx,Ly):    
#    for index_out,val_out in range(centers_gathered_x.shape[0]):
#    print("Process "+str(mpirank)+": Entered heating routine")
    for index_out,val_out in np.ndenumerate(centers_gathered_x):
        xx              = val_out
        yy              = centers_gathered_y[index_out]
        time_convecting = centers_gathered_times[index_out]
        #if time_convecting > t:
        #    print("I have futures ")
#        print("Process "+str(mpirank)+": I replicated one center")
        centerandghosts = create_ghosts(xx,yy,R,Lx,Ly)
        for centerghosted in centerandghosts:
            xxx = centerghosted[0]
            yyy = centerghosted[1]
            if xxx > (xmax_local + R) or xxx < (xmin_local - R) or yyy < (ymin_local - R) or yyy > (ymax_local + R ):
                continue
            else:
                #print("Process "+str(mpirank)+": Heating around one center")
                for index_in,val_in in np.ndenumerate(x):
                    distsq = (val_in-xxx)**2+(y[index_in]-yyy)**2
                    if distsq <= R2:
                        Q[index_in] = Q[index_in] + heatingfunction(t,distsq,time_convecting,q0,tauc,R2)    
    return None


"""
heat_mpi2(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local)

Receives the positions of the centers of convection and fills the matrix Q with the convective heating terms.

Q is an array of the same shape as h
x,y are arrays containing the coordinates of the portion of the domain that is local to this mpi process
t is an scalar with the current time
centers_gathered_x,centers_gathered_y and centers_gathered_times are arrays of the cordinates of the convecting points and the times at which they started convecting
q0, tauc, R2, R are floats with the parameters for the convection scheme
xmin_local, xmax_local, ymin_local, ymax_local are floats with the limits of the local portion of the domain

This version considers periodic boundaries by calling the ghosts routine with numpy arrays

"""

@jit(nopython=True,cache=False)
def heat_mpi2(Q,x,y,t,centers_gathered_x,centers_gathered_y,centers_gathered_times,q0,tauc,R2,R,xmin_local,xmax_local,ymin_local,ymax_local,Lx,Ly):    
#    for index_out,val_out in range(centers_gathered_x.shape[0]):
    for index_out,val_out in np.ndenumerate(centers_gathered_x):
        xx              = val_out
        yy              = centers_gathered_y[index_out]
        time_convecting = centers_gathered_times[index_out]
        centerandghosts = create_ghosts2(xx,yy,R,Lx,Ly)
        for ind,val in enumerate(centerandghosts[0]):
            xxx = val
            yyy = centerandghosts[1][ind]
            if xxx > (xmax_local + R) or xxx < (xmin_local - R) or yyy < (ymin_local - R) or yyy > (ymax_local + R ):
                continue
            else:
                for index_in,val_in in np.ndenumerate(x):
                    distsq = (val_in-xxx)**2+(y[index_in]-yyy)**2
                    if distsq <= R2:
                        Q[index_in] = Q[index_in] + heatingfunction(t,distsq,time_convecting,q0,tauc,R2)    
    return None


"""
create_ghosts(x,y,R)
Receives the coordinates of the convective centers and radius of convection. 
Returns a list with the same center and, if the center is near a border, the corresponding centers in the other side of the domain to consider periodic boundary conditions.

x,y are floats with the position of the convecting poing
R is the radius of convection

Uses Lx, and Ly, global variables with the limits of the domain

"""

@jit(nopython=True,cache=False)
def create_ghosts(x,y,R,Lx,Ly):
    left_border = x < R
    right_border = x > (Lx - R)
    upper_border = y > (Ly - R)
    lower_border = y < R
    centers = [(x , y)]
    if left_border:
        if upper_border:
            centers.append((x + Lx, y))
            centers.append((x, y - Ly))
            centers.append((x + Lx,y - Ly))
        elif lower_border:
            centers.append((x + Lx, y))
            centers.append((x, y + Ly))
            centers.append((x + Lx,y + Ly))
        else:
            centers.append((x + Lx, y))
    elif right_border:
        if upper_border:
            centers.append((x - Lx, y))
            centers.append((x, y - Ly))
            centers.append((x - Lx,y - Ly))
        elif lower_border:
            centers.append((x - Lx, y))
            centers.append((x, y + Ly))
            centers.append((x - Lx,y + Ly))
        else:
            centers.append((x - Lx, y))
    elif upper_border:
        centers.append((x, y - Ly))
    elif lower_border:
        centers.append((x,y + Ly))
    return centers

"""
create_ghosts2(x,y,R)
Receives the coordinates of the convective centers and radius of convection. 
Returns a numpy.array with the same center and, if the center is near a border, the corresponding centers in the other side of the domain to consider periodic boundary conditions.

x,y are floats with the position of the convecting poing
R is the radius of convection

Uses Lx, and Ly, global variables with the limits of the domain

"""

@jit(nopython=True,cache=False)
def create_ghosts2(x,y,R,Lx,Ly):
    left_border = x < R
    right_border = x > (Lx - R)
    upper_border = y > (Ly - R)
    lower_border = y < R
    if left_border:
        if upper_border:
            centers = np.array([ [x, x+Lx, x, x+Lx] , [y, y, y - Ly, y - Ly] ])
        elif lower_border:
            centers = np.array([ [x, x+Lx, x, x+Lx] , [y, y, y + Ly, y + Ly] ])
        else:
            centers = np.array([ [x, x+Lx] , [y, y] ])
    elif right_border:
        if upper_border:
            centers = np.array([ [x, x-Lx, x, x-Lx] , [y, y, y - Ly, y - Ly] ])
        elif lower_border:
            centers = np.array([ [x, x-Lx, x, x-Lx] , [y, y, y + Ly, y + Ly] ])
        else:
            centers = np.array([ [x, x-Lx] , [y, y] ])
    elif upper_border:
        centers = np.array([ [x, x], [y,y - Ly] ])
    elif lower_border:
        centers = np.array([ [x,x], [y,y+Ly] ])
    else:
        centers = np.array([[x],[y]])
        
    return centers

"""
Original version of heat_mpi, will probably die soon
"""

@jit(nopython=True,cache=False)
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

"""
computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times):
Fills the conv_centers and conv_centers times arrays to signal the convecting points and how long have they been convecting

"""
@jit(nopython=True,cache=False)
def computecentersandtimes(t,h,hc,tauc,conv_centers,conv_centers_times):
    """ This functions takes arrays """
    for ind,val in np.ndenumerate(h):
        if conv_centers[ind]:
            if ((t - conv_centers_times[ind]) < tauc):
                continue
            else:
                conv_centers[ind] = False
                conv_centers_times[ind] = 0.0
        elif not(conv_centers[ind]) and  (val < hc):
            conv_centers[ind] = True
            conv_centers_times[ind] = t
            # cct = conv_centers_times[ind]
            # if cct == 0.0 or ((t - cct) >= tauc) or (cct > t):
            #     conv_centers_times[ind] = t
            # else:
            #    continue        
    return None

"""
heatingfunction(t,distsq,conv_center_time,q0,tauc,R2):

Computes the magnitude of the heating according to Yang's paper.

t is the current time
distsq is the square of the distance of the point to the convecting point

conv_center_time is for how long has this time being convecting

q0, tauc, R2 are the parameters of the convection

"""

@jit(nopython=True,cache=False)
def heatingfunction(t,distsq,conv_center_time,q0,tauc,R2):
    """ This function must receive strictly floats """
    ##In RungeKutta sometimes the time evaluated is less than that in the previous timestep
    
    A0       = np.pi*R2
    deltat   = t - conv_center_time
    quotient = 2.0 * (deltat - tauc/2.0)/(tauc)
    q        = q0*(1.0 - quotient*quotient)*(1.0 - (distsq / R2))
    # if (q/ (tauc*A0) ) < 0:
    #     print("Heating is negative. This is not right.")
    # if deltat < 0.0:
    #     print("Delta t is negative")
    #     print("Current time: ",t)
    #     print("Conv center time",conv_center_time)
    return  q / (tauc*A0)
