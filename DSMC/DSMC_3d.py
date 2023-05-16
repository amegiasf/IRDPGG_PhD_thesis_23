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
def pair_collision(p1,p2,kappa,beta,alpha,vmax,vx,vy,vz,omegax,omegay,omegaz):
    kk = kappa/(1.+kappa)
    velx = vx[p1] - vx[p2]
    vely = vy[p1] - vy[p2]
    velz = vz[p1] - vz[p2]
    omx = (omegax[p1] + omegax[p2])*0.5
    omy = (omegay[p1] + omegay[p2])*0.5
    omz = (omegaz[p1] + omegaz[p2])*0.5
    phi = np.random.uniform(0,2*pi)
    cos_theta = np.random.uniform(-1.,1.)
    sin_theta = sqrt(1.-cos_theta*cos_theta)
    ex = cos(phi)*sin_theta
    ey = sin(phi)*sin_theta
    ez = cos_theta
    eevv = ex*velx+ey*vely+ez*velz
    eeww = ex*omx+ey*omy+ez*omz
    sxSx = ey*omz-ez*omy
    sxSy = ez*omx-ex*omz
    sxSz = ex*omy-ey*omx 
    f1x = 0.5*(1+alpha)*eevv*ex
    f1y = 0.5*(1+alpha)*eevv*ey
    f1z = 0.5*(1+alpha)*eevv*ez
    f2x = 0.5*(1+beta)*(velx-eevv*ex-sxSx)
    f2y = 0.5*(1+beta)*(vely-eevv*ey-sxSy)
    f2z = 0.5*(1+beta)*(velz-eevv*ez-sxSz)
    factorx = f1x+kk*f2x
    factory = f1y+kk*f2y
    factorz = f1z+kk*f2z
    dice = np.random.uniform(0,vmax)
    if(abs(eevv) >= dice):
        factor_omegax = ((1.+beta)/(SIGMA*(1.+kappa)))*(ey*velz-ez*vely+omx-eeww*ex)
        factor_omegay = ((1.+beta)/(SIGMA*(1.+kappa)))*(ez*velx-ex*velz+omy-eeww*ey)
        factor_omegaz = ((1.+beta)/(SIGMA*(1.+kappa)))*(ex*vely-ey*velx+omz-eeww*ez)
        vx[p1] -= factorx
        vx[p2] += factorx
        vy[p1] -= factory
        vy[p2] += factory
        vz[p1] -= factorz
        vz[p2] += factorz
        omegax[p1] -= factor_omegax
        omegax[p2] -= factor_omegax
        omegay[p1] -= factor_omegay
        omegay[p2] -= factor_omegay
        omegaz[p1] -= factor_omegaz
        omegaz[p2] -= factor_omegaz
        add = 1
        modp1 = sqrt(vx[p1]*vx[p1]+vy[p1]*vy[p1]+vz[p1]*vz[p1])
        modp2 = sqrt(vx[p2]*vx[p2]+vy[p2]*vy[p2]+vz[p2]*vz[p2])
        if(modp1>=vmax):
            vmax = modp1
        elif(modp2>=vmax):
            vmax = modp2
    else:
        add = 0
    return add, vmax, vx,vy,vz,omegax,omegay,omegaz

def print_results(cont,vx,vy,vz,omegax,omegay,omegaz,alpha,beta,folder1,file2,resc_t,aleat):
    temp_t = Temp(vx,vy,vz)
    temp_r = Temp_rot(omegax,omegay,omegaz)
    ff = folder1+'out_'+str(alpha)+'_'+str(beta)+'_'+str(cont/NPART)+'_'+aleat+'.txt'
    cx = vx/sqrt(2*temp_t/MASS)
    cy = vy/sqrt(2*temp_t/MASS)
    cz = vz/sqrt(2*temp_t/MASS)
    wx = omegax/sqrt(2*temp_r/INERTIA)
    wy = omegay/sqrt(2*temp_r/INERTIA)
    wz = omegaz/sqrt(2*temp_r/INERTIA)
    np.savetxt(ff,np.array([cx,cy,cz,wx,wy,wz]).T,delimiter='\t')
    file2.write(str(cont)+'\t'+str(temp_t/(resc_t*resc_t))+'\t'+str(temp_r/(resc_t*resc_t))+'\t'+str(temp_r/temp_t)+'\n')



#Subroutine to sample the collisions
def collisions(NCPP_PRINT,NCPP,ncollisions,dcollrest,vmax,DT,alpha,beta,kappa,vx,vy,vz,omegax,omegay,omegaz,folder,file2,resc_t,aleat):
    ncols = ncollisions
    nnp = NPART
    if(nnp>1):
        dcoll = CC1*nnp*(nnp-1)*5*vmax*DT+dcollrest
        ncoll = int(dcoll)
        dd = dcoll - ncoll
        for _ in range(ncoll):
            index1 = np.random.randint(0,nnp)
            index2 = (index1+np.random.randint(0,nnp-1)+1)%nnp
            p1 = index1
            p2 = index2
            coll, vmax,vx,vy,vz,omegax,omegay,omegaz = pair_collision(p1,p2,kappa,beta,alpha,vmax,vx,vy,vz,omegax,omegay,omegaz)
            ncols += coll
            if(coll ==1 and ncols<=25*NPART and ncols%int(0.2*NPART)==0):
                print_results(ncols,vx,vy,vz,omegax,omegay,omegaz,alpha,beta,folder,file2,resc_t,aleat)
            elif(coll ==1 and ncols%NCPP_PRINT == 0):
                print_results(ncols,vx,vy,vz,omegax,omegay,omegaz,alpha,beta,folder,file2,resc_t,aleat)
            if(ncols == NCPP):
                return ncols, vmax,vx,vy,vz,omegax,omegay,omegaz,dd 
    return ncols,vmax,vx,vy,vz,omegax,omegay,omegaz,dd
#Initialize positions and velocities
def initial_pos_vel(vx,vy,vz,omegax,omegay,omegaz):
    np.random.seed()
    vx = vx - np.sum(vx)/float(NPART)
    vy = vy - np.sum(vy)/float(NPART)
    vz = vz - np.sum(vz)/float(NPART)
    omegax = omegax - np.sum(omegax)/float(NPART)
    omegay = omegay - np.sum(omegay)/float(NPART)
    omegaz = omegaz - np.sum(omegaz)/float(NPART)
    suma = np.sum(vx*vx+vy*vy+vz*vz)
    suma2 = np.sum(omegax*omegax+omegay*omegay+omegaz*omegaz)
    scale = 1./sqrt(suma*MASS/(3.*NPART))
    scale2 = 1./sqrt(suma2*MASS*KAPPA*SIGMA*SIGMA*0.25/(3.*NPART))
    vx = scale*vx
    vy = scale*vy
    vz = scale*vz
    omegax = scale2*omegax
    omegay = scale2*omegay
    omegaz = scale2*omegaz
    vmax = np.max(vx*vx+vy*vy+vz*vz)
    return vmax,vx,vy,vz,omegax,omegay,omegaz

#Ballistic motion of the particles
def propagate(DT,noise,vx,vy,vz):
    if(noise):
        wx = np.random.standard_normal()
        wy = np.random.standard_normal()
        wz = np.random.standard_normal()
        vx = vx + sqrt(CHI0*DT)*wx
        vy = vy + sqrt(CHI0*DT)*wy
        vz = vz + sqrt(CHI0*DT)*wz
    return vx,vy,vz


#Temperature computation
def Temp(vx,vy,vz):
    vel2 = np.sum(vx*vx+vy*vy+vz*vz)
    temp = MASS*vel2/(3.*float(NPART))
    return temp

def Temp_rot(omegax,omegay,omegaz):
    omega2 = np.sum(omegax*omegax+omegay*omegay+omegaz*omegaz)
    temp_r = MASS*KAPPA*SIGMA*SIGMA*0.25*omega2/(3.*float(NPART))
    return temp_r

#Rescaling of the particle velocities
def rescaling(temp_t,vx,vy,vz,omegax,omegay,omegaz):
    scale_t = 1./sqrt(temp_t)
    vx = scale_t*vx
    vy = scale_t*vy
    vz = scale_t*vz
    omegax = scale_t*omegax
    omegay = scale_t*omegay
    omegaz = scale_t*omegaz
    return scale_t,vx,vy,vz,omegax,omegay,omegaz
#    print(Temp(),Temp_rot())

def rescaling_rot(temp:float,omegax,omegay,omegaz):
    print(temp)
    scale = 1./sqrt(temp)
    for i in range(NPART):
        omegax[i] = scale*omegax[i]
        omegay[i] = scale*omegay[i]
        omegaz[i] = scale*omegaz[i]
    return scale

#Frequency computation
def frequency(temp:float):
    return 4*NN*SIGMA*SIGMA*sqrt(pi*temp/MASS)

#White-noise. Langevin-type thermostat implemented


#Running the hole DSMC method
def run(NCPP,alpha,beta,kappa,folder,index,noise,vx,vy,vz,omegax,omegay,omegaz,cont):
    index = np.random.randint(0,NPART)
    aleat = str(abs(int(vx[index]*1.e10)))
    print(beta,aleat)
    #f.write(str(beta)+'\t'+str(aleat)+'\n')
    file2 = open(folder+str(alpha)+'_'+str(beta)+'_Temperatures_'+aleat+'.txt','tw')
    ncollisions = 0
    resc_t = 1.
    vmax,vx,vy,vz,omegax,omegay,omegaz = initial_pos_vel(vx,vy,vz,omegax,omegay,omegaz)
    print_results(ncollisions,vx,vy,vz,omegax,omegay,omegaz,alpha,beta,folder,file2,resc_t,aleat)
    tt = Temp(vx,vy,vz)
    dcollrest = 0.
    while(ncollisions < NCPP):
        DT = LBOX/vmax
        if(noise):
            vx,vy,vz = propagate(DT,noise,vx,vy,vz)
        coll, vmax,vx,vy,vz,omegax,omegay,omegaz,dcollrest = collisions(NCPP_PRINT,NCPP,ncollisions,dcollrest,vmax,DT,alpha,beta,kappa,vx,vy,vz,omegax,omegay,omegaz,folder,file2,resc_t,aleat)
        ncollisions = coll
        tt = Temp(vx,vy,vz)
        if(tt<0.7):
            r,vx,vy,vz,omegax,omegay,omegaz = rescaling(tt,vx,vy,vz,omegax,omegay,omegaz)
            resc_t *= r
    file2.close()

        

'''
Initialization of constants
'''
NPART = 100000 #Number of particles of the system
MASS = 1. #Mass of the particles
SIGMA = 1. #Size of the particles
ALPHA = 0.9 #Normal Coefficient of Restitution
#BETA = -0.575 #Tangential Coefficient of Restitution
KAPPA = 2./5. #Reduced moment of inertia
INERTIA = KAPPA*MASS*SIGMA*SIGMA*0.25
LONG = 4254.3 #Size of the SYSTEM-BOX
VOL = LONG*LONG*LONG #Volume of the SYSTEM-BOX
VOLI = 1./VOL #Inverse volume
NN = NPART*VOLI #Density of particles
NBOX = 1 #Number of subboxes
LBOX = LONG/NBOX #Length of subboxes
CC1 = pi*SIGMA*SIGMA*NBOX*NBOX*NBOX*VOLI #Constant of the method, for sampling collisions
FREQ = 1 #Number of iterations for saving data
FRES = int(NPART**2) #Number of collisions to rescale velocities
NITERATES = 150 #Number of iterations
NPP = 150 #Total number of collision per particle
NPP_PRINT = 2 #Number of collisions per particle one wants to print the results
NCPP_PRINT = int(NPP_PRINT*NPART) #Total number of collisions one wants to print the results
NCPP = NPP*NPART #Total number of collisions
ZETA = (1-ALPHA*ALPHA)/3. #Cooling rate for the smooth case
ENSEMBLES = 1 #Number of ensembles to be averaged
CHI0 = 24.*pi*NN/5.
noise = False
'''
Initialization of variables and main program.
vx, vy, vz: particle velocities
posx, posy, posz: particle positions
boxes: boxes three-dimensional array
dcollrest: remaining decimal part of dcoll-ncoll in the Particle sampling
'''
beta_init = -0.15
betas = [beta_init]
#f = open('/home/alberto/DOCTORADO/PRIMERO/simulations_DSMC/indices1.txt','tw')
for beta in betas:
    folder2 = '0.9_-0.15/'
    #folder2 = '/home/alberto/DOCTORADO/PRIMERO/simulations_DSMC/'
    #os.mkdir(folder2+str(ALPHA)+'_'+str(beta))
    #folder2 = folder2+str(ALPHA)+'_'+str(beta)+'/'
    for i in range(ENSEMBLES):
        np.random.seed()
        vx = np.random.standard_normal(NPART)
        vy = np.random.standard_normal(NPART)
        vz = np.random.standard_normal(NPART)
        omegax = np.random.standard_normal(NPART)
        omegay = np.random.standard_normal(NPART)
        omegaz = np.random.standard_normal(NPART)
        run(NCPP,ALPHA,beta,KAPPA,folder2,i,noise,vx,vy,vz,omegax,omegay,omegaz,i)
#f.close()
