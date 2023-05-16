"""
Alberto Megías Fernández. April 22nd, 2020.

This code correponds to a first attempt of DSMC method for monocomponent inelastic and rough granular gases.
The first version is in three-dimensions.
The organization of the code is the following:

- Importation of neccesary Python packages
- Definition of registers or other classes of variables needed in the code. In the first version, the 
register called Box is created.
- Functions, subroutines and procedures to divide to core of the code, and that are essential for DSMC.
- 'Main' program:
    - Constants of the system are initialized:
        + Number of particles
        + Mass and size of the particles
        + Coefficient of Restitution
        + Size of the Box
    - Constants of the method are initialized
    - The method consist in:
        1. Initializing the system random and homogeneously
        2. Particles move in a ballistic way, taking into account periodic boundary conditions. If noise == True,
        stochastic thermostat with white noise will be applied.
        3. After a DT time determined by the Temperature of the system, collisions are sampled.
        4. Only pairwise collisions are taken into account.
        5. A collision occurs or not comparing the collisional rule with a random number following
        a uniform distribution between 0 and a 'vmax'-maximum velocity of the particles-. The stochastic
        behavior of the method is this.
        6. Collisions are updated.
        7. If a number of collisions is reached (FRES), particle velocities are rescaled.
        8. Return to 2 untill NITERATIONS is reached.
    - The method is run ENSEMBLE-times in order to be averaged (even one can ignore average and, results are
    good enough running once).
- Outputs are shown 
"""
from math import sqrt,pi,exp,cos,sin,fmod
import numpy as np
import os
"""
Some precedures, functions and subroutines, to make the code easier.
"""
#Computation if a pair collision is done or not
def pair_collision(p1,p2,kappa,beta,alpha,sigma,vmax,vx,vy,omegaz):
    kk = kappa/(1.+kappa)
    velx = vx[p1] - vx[p2]
    vely = vy[p1] - vy[p2]
    omz = (omegaz[p1] + omegaz[p2])*0.5*sigma
    x = np.random.uniform(0,2*np.pi)
    ex = cos(x)
    ey = sin(x)
    eevv = ex*velx+ey*vely
    dice = np.random.uniform(0,vmax)
    if(abs(eevv) >= dice):
        f1x = 0.5*(1+alpha)*eevv*ex
        f1y = 0.5*(1+alpha)*eevv*ey
        f2x = 0.5*(1+beta)*(velx-eevv*ex-omz*ey)
        f2y = 0.5*(1+beta)*(vely-eevv*ey+omz*ex)
        factorx = f1x+kk*f2x
        factory = f1y+kk*f2y
        factor_omegaz = ((1.+beta)/(sigma*(1.+kappa)))*(ex*vely-ey*velx+omz)
        vx[p1] -= factorx
        vx[p2] += factorx
        vy[p1] -= factory
        vy[p2] += factory
        omegaz[p1] -= factor_omegaz
        omegaz[p2] -= factor_omegaz
        add = 1
        modp1 = sqrt(vx[p1]*vx[p1]+vy[p1]*vy[p1])
        modp2 = sqrt(vx[p2]*vx[p2]+vy[p2]*vy[p2])
        if(modp1>=vmax):
            vmax = modp1
        elif(modp2>=vmax):
            vmax = modp2
    else:
        add = 0
    return add, vmax, vx,vy,omegaz

def print_results(cont,vx,vy,omegaz,alpha,beta,folder1,file2,resc_t,aleat):
    temp_t = Temp(vx,vy)
    temp_r = Temp_rot(omegaz)
    ff = folder1+'out_2D_'+str(alpha)+'_'+str(beta)+'_'+str(cont/NPART)+'_'+aleat+'.txt'
    cx = vx/sqrt(2*temp_t/MASS)
    cy = vy/sqrt(2*temp_t/MASS)
    wz = omegaz/sqrt(2*temp_r/INERTIA)
    np.savetxt(ff,np.array([cx,cy,wz]).T,delimiter='\t')
    #print(temp_r/temp_t)
    file2.write(str(cont)+'\t'+str(temp_t/(resc_t*resc_t))+'\t'+str(temp_r/(resc_t*resc_t))+'\t'+str(temp_r/temp_t)+'\n')


#Subroutine to sample the collisions
def collisions(NCPP_PRINT,NCPP,ncollisions,vmax,alpha,beta,kappa,sigma,vx,vy,omegaz,folder,file2,resc_t,aleat,noise,CC1,DT,dcollrest):
    ncols = ncollisions
    nnp = NPART
    dcoll = CC1*vmax*DT+dcollrest
    ncoll = int(dcoll)
    dd = dcoll - ncoll
    for _ in range(ncoll):
        index1 = np.random.randint(0,nnp)
        index2 = (index1+np.random.randint(0,nnp-1)+1)%nnp
        p1 = index1
        p2 = index2
        coll, vmax,vx,vy,omegaz = pair_collision(p1,p2,kappa,beta,alpha,sigma,vmax,vx,vy,omegaz)
        ncols += coll
        if(coll ==1 and ncols<NCPP and ncols%int(0.5*NPART)==0):
            print_results(ncols,vx,vy,omegaz,alpha,beta,folder,file2,resc_t,aleat)
       # if(coll==1 and ncols%int(0.1*NPART)==0):
       #     vx = vx-np.average(vx)
       #     vy = vy-np.average(vy)
       #     omegaz = omegaz -np.average(omegaz)
       # elif(coll ==1 and ncols%NCPP_PRINT == 0):
       #     print_results(ncols,vx,vy,omegaz,alpha,beta,folder,file2,resc_t,aleat)
        if(coll == 1 and ncols%int(0.5*NPART)==0 and (not noise)):
            tt = Temp_tot(vx,vy,omegaz)
            r,vx,vy,omegaz = rescaling(tt,vx,vy,omegaz)
            resc_t *= r
            #print(np.average(vx),np.average(vy),np.average(omegaz))
        if(ncols == NCPP):
            return ncols, vmax,vx,vy,omegaz,dd
       # vx = vx-np.average(vx)
       # vy = vy-np.average(vy)
       # omegaz = omegaz -np.average(omegaz)
    return ncols,vmax,vx,vy,omegaz,dd
#Initialize positions and velocities
def initial_pos_vel(vx,vy,omegaz):
    np.random.seed()
    vx = vx - np.average(vx)
    vy = vy - np.average(vy)
    omegaz = omegaz - np.average(omegaz)
    suma = np.average(vx*vx+vy*vy)
    suma2 = np.average(omegaz*omegaz)
    scale = 1./sqrt(suma*MASS*0.5)
    scale2 = 1./sqrt(suma2*INERTIA)
    vx = scale*vx
    vy = scale*vy
    omegaz = scale2*omegaz
    #print(np.average(omegaz))
    vmax = np.max(np.sqrt(vx*vx+vy*vy))
    return vmax,vx,vy,omegaz

#Ballistic motion of the particles
#White-noise. Langevin-type thermostat implemented
def propagate(DT,noise,noise_type,vx,vy,omegaz,epsilon,MASS,INERTIA,CHI0,NPART):
    np.random.seed()
    if(noise):
        if(noise_type==1):
            wx = np.random.standard_normal(NPART)
            wy = np.random.standard_normal(NPART)
            wx -= np.average(wx)
            wy -= np.average(wy)
            w2mod = np.average(wx**2+wy**2)
            wx /= np.sqrt(w2mod/2)
            wy /= np.sqrt(w2mod/2)
            vx = vx + sqrt(CHI0*DT)*wx
            vy = vy + sqrt(CHI0*DT)*wy
        elif(noise_type==2):
            wz = np.random.standard_normal(NPART)
            wz -= np.average(wz)
            wz /= np.sqrt(np.average(wz**2))
            omegaz = omegaz + sqrt(CHI0*MASS*2*DT/INERTIA)*wz
        elif(noise_type==3):
            wx = np.random.standard_normal(NPART)
            wy = np.random.standard_normal(NPART)
            wx -= np.average(wx)
            wy -= np.average(wy)
            w2mod = np.average(wx**2+wy**2)
            wx /= np.sqrt(w2mod/2)
            wy /= np.sqrt(w2mod/2)
            wz = np.random.standard_normal(NPART)
            wz -= np.average(wz)
            wz /= np.sqrt(np.average(wz**2))
            omegaz = omegaz + sqrt(CHI0*epsilon*MASS*2*DT/INERTIA)*wz
            vx = vx + sqrt(CHI0*(1-epsilon)*DT)*wx
            vy = vy + sqrt(CHI0*(1-epsilon)*DT)*wy
    return vx,vy,omegaz


#Temperature computation
def Temp(vx,vy):
    vel2 = np.average(vx*vx+vy*vy)
    temp = MASS*vel2*0.5
    return temp

def Temp_rot(omegaz):
    omega2 = np.average(omegaz*omegaz)
    temp_r = INERTIA*omega2
    return temp_r
def Temp_tot(vx,vy,omega):
    return (2*Temp(vx,vy)+Temp_rot(omega))/3.
#Rescaling of the particle velocities
def rescaling(temp_t,vx,vy,omegaz):
    scale_t = 1./sqrt(temp_t)
    vx = scale_t*vx
    vy = scale_t*vy
    omegaz = scale_t*omegaz
    vx = vx-np.average(vx)
    vy = vy-np.average(vy)
    omegaz = omegaz -np.average(omegaz)
    return scale_t,vx,vy,omegaz
#    print(Temp(),Temp_rot())


#Running the hole DSMC method
def run(NCPP,NN,NPART,CC1,MULTI,CHI0,alpha,beta,kappa,MASS,INERTIA,SIGMA,folder,index,noise,noise_type,epsilon,vx,vy,omegaz,cont):
    index = np.random.randint(0,NPART)
    aleat = str(abs(int(vx[index]*1.e10)))
    print(alpha,beta,aleat)
#    f.write(str(beta)+'\t'+str(aleat)+'\n')
    file2 = open(folder+str(alpha)+'_'+str(beta)+'_Temperatures_'+aleat+'.txt','tw')
    ncollisions = 0
    resc_t = 1.
    vmax,vx,vy,omegaz = initial_pos_vel(vx,vy,omegaz)
    print_results(ncollisions,vx,vy,omegaz,alpha,beta,folder,file2,resc_t,aleat)
    tt = Temp(vx,vy)
    dd = 0.
    DT = 0.01
    while(ncollisions < NCPP):
       # if(noise):
        #    vx,vy,omegaz = propagate(DT,noise,noise_type,vx,vy,omegaz,epsilon,MASS,INERTIA,CHI0,NPART)
        coll, vmax,vx,vy,omegaz,dd = collisions(NCPP_PRINT,NCPP,ncollisions,vmax,alpha,beta,kappa,SIGMA,vx,vy,omegaz,folder,file2,resc_t,aleat,noise,CC1,DT,dd)
        ncollisions = coll
        if(noise):
            vx,vy,omegaz = propagate(DT,noise,noise_type,vx,vy,omegaz,epsilon,MASS,INERTIA,CHI0,NPART)
            vmax = np.max(np.sqrt(vx**2+vy**2))
    file2.close()

        

'''
Initialization of constants
'''
NPART = 10000 #Number of particles of the system
MASS = 1. #Mass of the particles
SIGMA = 1. #Size of the particles
ALPHA = 0.9 #Normal Coefficient of Restitution
#BETA = -0.575 #Tangential Coefficient of Restitution
KAPPA = 1./2. #Reduced moment of inertia
INERTIA = KAPPA*MASS*SIGMA*SIGMA*0.25
LONG = 4254.3 #Size of the SYSTEM-BOX
VOL = LONG*LONG #Volume of the SYSTEM-BOX
VOLI = 1./VOL #Inverse volume
NN = NPART*VOLI #Density of particles
NBOX = 1 #Number of subboxes
LBOX = LONG/NBOX #Length of subboxes
CC1 = 0.5*np.pi*SIGMA*NPART*NN #Constant of the method, for sampling collisions
FREQ = 1 #Number of iterations for saving data
FRES = int(NPART**2) #Number of collisions to rescale velocities
NITERATES = 200 #Number of iterations
NPP = 150 #Total number of collision per particle
NPP_PRINT = 0.5 #Number of collisions per particle one wants to print the results
NCPP_PRINT = int(NPP_PRINT*NPART) #Total number of collisions one wants to print the results
NCPP = NPP*NPART #Total number of collisions
ENSEMBLES = 1 #Number of ensembles to be averaged
MULTI = 1
CHI0 = np.sqrt(MULTI**3)*24.*pi*NN/5.
noise = True
noise_type = 2
epsilon = 1
'''
Initialization of variables and main program.
vx, vy: particle velocities
omegaz: angular velocity
posx, posy, posz: particle positions
boxes: boxes three-dimensional array
dcollrest: remaining decimal part of dcoll-ncoll in the Particle sampling
'''
beta_init = 1
#betas = [0]
betas = np.arange(-0.95,0.0,0.05)
betas = np.array([float(format(b,'.2')) for b in betas])
#betas = [beta_init-i*0.025 for i in range(0)]
#f = open('/home/alberto/DOCTORADO/PRIMERO/simulations_DSMC/2D/indices_1.txt','tw')
for beta in betas:
    folder2 = 'NOISE_2/'
    #print(str(ALPHA)+'_'+str(beta)+'\n')
    #folder2 = str(ALPHA)+'_'+str(beta)+'/'
    for i in range(ENSEMBLES):
        np.random.seed()
        vx = np.random.standard_normal(NPART)
        vy = np.random.standard_normal(NPART)
        omegaz = np.random.standard_normal(NPART)
        run(NCPP,NN,NPART,CC1,CHI0,MULTI,ALPHA,beta,KAPPA,MASS,INERTIA,SIGMA,folder2,i,noise,noise_type,epsilon,vx,vy,omegaz,i)

