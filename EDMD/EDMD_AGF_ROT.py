'''
Alberto Megías Fernández. 10th December 2020.
Library for EDMD for granular gases with Langevin Dynamics via AGF in 2D, using next neighbor lists.
'''
import numpy as np
import copy
from math import pi,sqrt,cos,sin,fmod,exp
import heapq as pq
import matplotlib.pyplot as plt
import scipy.special as sc
#from scipy.stats import binom
#import mpl_scatter_density # adds projection='scatter_density'

def identify_cells(x,y,dd,cells,NN,NPART,LBOX,LCELLS):
    if(np.any(x<0) or np.any(x>LBOX)):
        x[x<0] += LBOX
        x[x>LBOX] -= LBOX
    if(np.any(y<0) or np.any(y>LBOX)):
        y[y<0] += LBOX
        y[y>LBOX] -= LBOX
    cont = 0
    for _ in range(NN):
        for _ in range(NN):
            if(cont<NPART):
                cellx = int(x[cont]//LCELLS)
                celly = int(y[cont]//LCELLS)
                cells[cellx][celly].append(cont)
                dd[cont] = (cellx,celly)
                cont += 1
    return dd,cells

def initial_rot(dd,x,y,vx,vy,w,cells,NPART,NNCELLS,LCELLS,SIGMA,NULL,MASS,INERTIA,TINIT,T_ROT_INIT):
    cont = 0
    vcomx = 0.
    vcomy = 0.
    NN = NNCELLS
    DL = LCELLS
    vx = np.random.standard_normal(NPART)
    vy = np.random.standard_normal(NPART)
    w = np.random.standard_normal(NPART)
    if(DL<=SIGMA):
        print('ERROR: OVERLAP FORCED')
    for i in range(NN):
        for j in range(NN):
            if(cont<NPART):
                x[cont] = np.random.uniform(NULL+(i)*DL+SIGMA,-NULL+(i+1)*DL-SIGMA)
                y[cont] = np.random.uniform(NULL+(j)*DL+SIGMA,-NULL+(j+1)*DL-SIGMA)
                cellx = int(x[cont]//LCELLS)
                celly = int(y[cont]//LCELLS)
                cells[cellx][celly].append(cont)
                dd[cont] = (cellx,celly)
                cont += 1
    vcomx = np.sum(vx)/float(NPART)
    vcomy = np.sum(vy)/float(NPART)
    w -= np.average(w)
    w /= np.sqrt(np.average(w**2)*INERTIA)
    w *= np.sqrt(T_ROT_INIT)
    vx,vy = vel_res(vcomx,vcomy,vx,vy,MASS,NPART)
    vx *= np.sqrt(TINIT)
    vy *= np.sqrt(TINIT)
    return x,y,vx,vy,w,dd,cells

def initial(dd,x,y,vx,vy,cells,NPART,NNCELLS,LCELLS,SIGMA,NULL,MASS,TINIT):
    cont = 0
    vcomx = 0.
    vcomy = 0.
    NN = NNCELLS
    DL = LCELLS
    vx = np.random.standard_normal(NPART)
    vy = np.random.standard_normal(NPART)
    if(DL<=SIGMA):
        print('ERROR: OVERLAP FORCED')
    for i in range(NN):
        for j in range(NN):
            if(cont<NPART):
                x[cont] = np.random.uniform(NULL+(i)*DL+SIGMA,-NULL+(i+1)*DL-SIGMA)
                y[cont] = np.random.uniform(NULL+(j)*DL+SIGMA,-NULL+(j+1)*DL-SIGMA)
                cellx = int(x[cont]//LCELLS)
                celly = int(y[cont]//LCELLS)
                cells[cellx][celly].append(cont)
                dd[cont] = (cellx,celly)
                cont += 1
    vcomx = np.sum(vx)/float(NPART)
    vcomy = np.sum(vy)/float(NPART)
    vx,vy = vel_res(vcomx,vcomy,vx,vy,MASS,NPART)
    vx *= np.sqrt(TINIT)
    vy *= np.sqrt(TINIT)
    return x,y,vx,vy

def a20_cumulant(vx,vy,temp,MASS,DIMENSION_T):
    cx = (vx-np.average(vx))/np.sqrt(2*temp/MASS)
    cy = (vy-np.average(vy))/np.sqrt(2*temp/MASS)
    c4 = np.average((cx**2+cy**2)**2)
    a2 = 4*c4/(DIMENSION_T*(DIMENSION_T+2))-1
    return a2
    
def a02_cumulant(w,temp_r,INERTIA,DIMENSION_R):
    ww = (w)/np.sqrt(2*temp_r/INERTIA)
    ww4 = np.average(ww**4)
    a02 = 4*ww4/(DIMENSION_R*(DIMENSION_R+2))-1
    return a02

def a11_cumulant(vx,vy,w,temp,temp_r,MASS,INERTIA,DIMENSION_T,DIMENSION_R):
    cx = (vx-np.average(vx))/np.sqrt(2*temp/MASS)
    cy = (vy-np.average(vy))/np.sqrt(2*temp/MASS)
    ww = (w)/np.sqrt(2*temp_r/INERTIA)
    c2w2 = (cx**2+cy**2)*ww**2
    a11 = 4*np.average(c2w2)/(DIMENSION_R*DIMENSION_T)-1
    return a11

def res_momentum_v2(vx,vy,vcomx,vcomy):
    vxx = vx-vcomx
    vyy = vy-vcomy
    return vx-vcomx, vy-vcomy, np.sum(vxx*vxx+vyy*vyy)

def rescale(scale,vx,vy):
    return scale*vx, scale*vy

def rescale_rot(vx,vy,w,MASS,INERTIA): #RESCALING TO T_TOT = (2 T_T+T_R)/(3) = 1
    scale = Temperature(vx,vy,MASS)
    scale = 1/np.sqrt(scale)
    #vx -= np.average(vx)
    #vy -= np.average(vy)
    #w -= np.average(w)
    return scale*vx, scale*vy, scale*w, scale

def Temperature(vx,vy,MASS):
    return MASS*np.average((vx-np.average(vx))**2+(vy-np.average(vy))**2)/(2.)

def Temp_rot(w,INERTIA):
    w_av = np.average((w)**2)
    temp_r = INERTIA*w_av
    return temp_r

def Temp_tot(vx,vy,w,MASS,INERTIA):
    return (2*Temperature(vx,vy,MASS)+Temp_rot(w,INERTIA))/3.

def vel_res(vcomx,vcomy,vx,vy,MASS,NPART):
    vx,vy,v2 = res_momentum_v2(vx,vy,vcomx,vcomy)
    scale = sqrt(v2*MASS*0.5/float(NPART))
    scale = 1./scale
    vx,vy = rescale(scale,vx,vy)
    return vx,vy

def propagation(x,y,vx,vy,tprop,LBOX):
    return np.fmod(x +tprop*vx,LBOX), np.fmod(y +tprop*vy,LBOX)


def langevin_propagation(x,y,vx,vy,w,tprop,NPART,DIMENSION,GAMMA,ZETA0,EPSILON,TBATH,MASS,INERTIA,NOISE_TYPE,LBOX):
    if(NOISE_TYPE == 1): #LAGENVIN-TYPE: CONSTANT DRAG + NOISE (LINKED BY FLUCTUATION-DISSIPATION THEOREM)
        y0x = np.random.standard_normal(NPART)
        y0y = np.random.standard_normal(NPART)
        y1x = np.random.standard_normal(NPART)
        y1y = np.random.standard_normal(NPART)
        y0x -= np.mean(y0x)
        y0y -= np.mean(y0y)
        y1x -= np.mean(y1x)
        y1y -= np.mean(y1y)
        y02 = np.average(y0x**2+y0y**2)/DIMENSION
        y12 = np.average(y1x**2+y1y**2)/DIMENSION
        y0x /= np.sqrt(y02)
        y0y /= np.sqrt(y02)
        y1x /= np.sqrt(y12)
        y1y /= np.sqrt(y12)
        #y0x,y0y = vel_res(np.average(y0x),np.average(y0y),y0x,y0y,MASS,NPART)
        #y1x,y1y = vel_res(np.average(y1x),np.average(y1y),y1x,y1y,MASS,NPART)
        w02 = 0.5*(1-np.exp(-2*GAMMA*tprop))/GAMMA
        w0x = np.sqrt(w02)*y0x
        w0y = np.sqrt(w02)*y0y
        w1w0 =  0.5*(1-np.exp(-GAMMA*tprop))**2/GAMMA**2
        w12 =  0.5*(2*tprop-(3+np.exp(-2*GAMMA*tprop)-4*np.exp(-GAMMA*tprop))/GAMMA)/GAMMA**2
        w1x = (w1w0/np.sqrt(w02))*y0x+np.sqrt(w12-w1w0**2/w02)*y1x
        w1y = (w1w0/np.sqrt(w02))*y0y+np.sqrt(w12-w1w0**2/w02)*y1y
        vvx = vx*np.exp(-GAMMA*tprop)+np.sqrt(2*TBATH*GAMMA)*w0x
        vvy = vy*np.exp(-GAMMA*tprop)+np.sqrt(2*TBATH*GAMMA)*w0y
        rrx = x+(1-np.exp(-GAMMA*tprop))*vx/GAMMA+np.sqrt(2*TBATH*GAMMA)*w1x
        rry = y+(1-np.exp(-GAMMA*tprop))*vy/GAMMA+np.sqrt(2*TBATH*GAMMA)*w1y
        wwz = w
    elif(NOISE_TYPE == 2): #STOCHASTIC THERMOSTAT (\chi_0 <-- GAMMA)
        y0x = np.random.standard_normal(NPART)
        y0y = np.random.standard_normal(NPART)
        y1x = np.random.standard_normal(NPART)
        y1y = np.random.standard_normal(NPART)
        y0x -= np.mean(y0x)
        y0y -= np.mean(y0y)
        y1x -= np.mean(y1x)
        y1y -= np.mean(y1y)
        y02 = np.average(y0x**2+y0y**2)/DIMENSION
        y12 = np.average(y1x**2+y1y**2)/DIMENSION
        y0x /= np.sqrt(y02)
        y0y /= np.sqrt(y02)
        y1x /= np.sqrt(y12)
        y1y /= np.sqrt(y12)
        w02 = GAMMA**2*tprop
        w0x = np.sqrt(w02)*y0x
        w0y = np.sqrt(w02)*y0y
        w1w0 =  0.5*GAMMA**2*tprop**2
        w12 =  (2.*GAMMA**2*tprop**3)/3.
        w1x = (w1w0/np.sqrt(w02))*y0x+np.sqrt(w12-w1w0**2/w02)*y1x
        w1y = (w1w0/np.sqrt(w02))*y0y+np.sqrt(w12-w1w0**2/w02)*y1y
        vvx = vx+w0x
        vvy = vy+w0y
        rrx = x+vx*tprop+w1x
        rry = y+vy*tprop+w1y
        wwz = w
    elif(NOISE_TYPE == 3): #STOCHASTIC ROTATIONAL THERMOSTAT (\chi_0 <-- GAMMA)
        CHIO_EFFECTIVE_ROT = GAMMA*np.sqrt(MASS*2/INERTIA)
        yrot = np.random.standard_normal(NPART)
        yrot -= np.mean(yrot)
        yrot /= np.sqrt(np.average(yrot**2))
        wwz = w + CHIO_EFFECTIVE_ROT*np.sqrt(tprop)*yrot
        rrx = x+vx*tprop
        rry = y+vy*tprop
        vvx = vx
        vvy = vy
    elif(NOISE_TYPE == 4): #STOCHASTIC SPLITTING-THERMOSTAT
        CHIO_EFFECTIVE_ROT = GAMMA*np.sqrt(EPSILON*MASS*2/INERTIA)
        CHIO_EFFECTIVE_TR = GAMMA*np.sqrt((1-EPSILON))
        if(EPSILON!=1):
            y0x = np.random.standard_normal(NPART)
            y0y = np.random.standard_normal(NPART)
            y1x = np.random.standard_normal(NPART)
            y1y = np.random.standard_normal(NPART)
            y0x -= np.mean(y0x)
            y0y -= np.mean(y0y)
            y1x -= np.mean(y1x)
            y1y -= np.mean(y1y)
            y02 = np.average(y0x**2+y0y**2)/2
            y12 = np.average(y1x**2+y1y**2)/2
            y0x /= np.sqrt(y02)
            y0y /= np.sqrt(y02)
            y1x /= np.sqrt(y12)
            y1y /= np.sqrt(y12)
            w02 = CHIO_EFFECTIVE_TR**2*tprop
            w0x = np.sqrt(w02)*y0x
            w0y = np.sqrt(w02)*y0y
            w1w0 =  0.5*CHIO_EFFECTIVE_TR**2*tprop**2
            w12 =  (2.*CHIO_EFFECTIVE_TR**2*tprop**3)/3.
            w1x = (w1w0/np.sqrt(w02))*y0x+np.sqrt(w12-w1w0**2/w02)*y1x
            w1y = (w1w0/np.sqrt(w02))*y0y+np.sqrt(w12-w1w0**2/w02)*y1y
        yrot = np.random.standard_normal(NPART)
        yrot -= np.mean(yrot)
        yrot /= np.sqrt(np.average(yrot**2))
        wwz = w + CHIO_EFFECTIVE_ROT*np.sqrt(tprop)*yrot
        if(EPSILON!=1):
            vvx = vx+w0x
            vvy = vy+w0y
            rrx = x+vx*tprop+w1x
            rry = y+vy*tprop+w1y
        else:
            vvx = vx
            vvy = vy
            rrx = x+vx*tprop
            rry = y+vy*tprop
    elif(NOISE_TYPE == 5): #QUADRATIC NONLINEAR DRAG CASE
        v2 = vx**2+vy**2
        zeta = zetaeff(v2,ZETA0,GAMMA,MASS,TBATH)
        xisquared = xi2(v2,ZETA0,GAMMA,MASS,TBATH)
        y0x = np.random.standard_normal(NPART)
        y0y = np.random.standard_normal(NPART)
        y1x = np.random.standard_normal(NPART)
        y1y = np.random.standard_normal(NPART)
        y0x -= np.mean(y0x)
        y0y -= np.mean(y0y)
        y1x -= np.mean(y1x)
        y1y -= np.mean(y1y)
        y02 = np.average(y0x**2+y0y**2)/DIMENSION
        y12 = np.average(y1x**2+y1y**2)/DIMENSION
        y0x /= np.sqrt(y02)
        y0y /= np.sqrt(y02)
        y1x /= np.sqrt(y12)
        y1y /= np.sqrt(y12)
        w02 = tprop
        w0x = np.sqrt(w02)*y0x
        w0y = np.sqrt(w02)*y0y
        w1w0 =  0.5*tprop**2
        w12 =  (2.*tprop**3)/3.
        w1x = (w1w0/np.sqrt(w02))*y0x+np.sqrt(w12-w1w0**2/w02)*y1x
        w1y = (w1w0/np.sqrt(w02))*y0y+np.sqrt(w12-w1w0**2/w02)*y1y
        vvx = vx*(1-zeta)+np.sqrt(xisquared)*w0x
        vvy = vy*(1-zeta)+np.sqrt(xisquared)*w0y
        rrx = x+vx*tprop+np.sqrt(xisquared)*w1x
        rry = y+vy*tprop+np.sqrt(xisquared)*w1y
        wwz = w
    elif(NOISE_TYPE == 0):
        rrx,rry = propagation(x,y,vx,vy,tprop,LBOX)
        vvx,vvy,wwz = vx,vy,w
    return rrx,rry,vvx,vvy,wwz

def zetaeff(v2,zeta0,gamma,m,kbT):
    return zeta0*(1-2*gamma+gamma*m*v2/kbT)

def xi2(v2,zeta0,gamma,m,kbT):
    return 2*kbT*zeta0*(1+gamma*m*v2/kbT)/m

def boundary(dx,dy,LHALF,LBOX):
    if(dx>LHALF):
        dx -= LBOX
    if(dx<-LHALF):
        dx += LBOX
    if(dy>LHALF):
        dy -= LBOX
    if(dy<-LHALF):
        dy += LBOX
    return dx,dy

def collision_rot(i,j,alpha,beta,dt,x,y,vx,vy,w,counter,LHALF,LBOX,LCELLS,SIGMA,KAPPA,TIMEC,NULL,NULL2,MASS,COLL_TYPE,flag):
    CT = COLL_TYPE
    if(flag):
        if(dt < 1.e-12*TIMEC):
            dt = 1.e-12*TIMEC
            CT = 1
    dvx = vx[i]-vx[j]
    dvy = vy[i]-vy[j]
    dx = x[i]-x[j]
    dy = y[i]-y[j]
    ncol = 0
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    scalar = dx*dvx+dy*dvy
    dxx = dx + dt*dvx
    dyy = dy +dt*dvy
    d2 = dxx*dxx+dyy*dyy
    RR = SIGMA
    ff = False
    if(abs(d2-RR*RR)>NULL2):
        if(dt==0):
            dt = NULL
        ff = True
        #x,y = propagation(x,y,vx,vy,dt,LBOX)
        return x,y,vx,vy,w,dt,ncol,counter,ff
    #    ncol = 1
    #    scalar  = dx*dvx+dy*dvy
    #    if(scalar<0):
    #        d2 = dx*dx+dy*dy
    #        v2 = dvx*dvx+dvy*dvy
    #        q = d2-RR*RR
    #        ww = scalar*scalar-q*v2
    #        if(ww<0):
    #            return x,y,vx,vy,w,dt,ncol,counter
            #print(counter[counter>1])
    #        dt = q/(-scalar+sqrt(ww))
    #    else:
    #        return x,y,vx,vy,w,dt,ncol,counter
    if(abs(d2-RR*RR)>NULL and abs(d2-RR*RR)<NULL2):
        scalar  = dx*dvx+dy*dvy
        if(scalar<0):
            d2 = dx*dx+dy*dy
            v2 = dvx*dvx+dvy*dvy
            q = d2-RR*RR
            ww = scalar*scalar-q*v2
            if(ww<0):
                return x,y,vx,vy,w,dt,ncol,counter,ff
            #print(counter[counter>1])
            dt = q/(-scalar+sqrt(ww))
        else:
            return x,y,vx,vy,w,dt,ncol,counter,ff
    x,y = propagation(x,y,vx,vy,dt,LBOX)
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    dist = sqrt(dx*dx+dy*dy)
    ndx = dx/dist
    ndy = dy/dist
    vx[i],vx[j],vy[i],vy[j],w[i],w[j] = collisional_rule(alpha,beta,ndx,ndy,dvx,dvy,vx[i],vx[j],vy[i],vy[j],w[i],w[j],SIGMA,KAPPA,MASS,CT)
    counter[i] += 1
    counter[j] += 1
    ncol = 1
    return x,y,vx,vy,w,dt,ncol,counter,ff

def collision(i,j,alpha,dt,x,y,vx,vy,counter,LHALF,LBOX,LCELLS,SIGMA,MASS,NULL,NULL2,COLL_TYPE):
    dvx = vx[i]-vx[j]
    dvy = vy[i]-vy[j]
    dx = x[i]-x[j]
    dy = y[i]-y[j]
    ncol = 0
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    scalar = dx*dvx+dy*dvy
    dxx = dx + dt*dvx
    dyy = dy +dt*dvy
    d2 = dxx*dxx+dyy*dyy
    RR = SIGMA
    if(abs(d2-RR*RR)>NULL2):
        print(np.sqrt(d2-RR*RR),LCELLS,dx,dy,scalar)
        return x,y,vx,vy,dt,ncol
    if(abs(d2-RR*RR)>NULL):
        scalar  = dx*dvx+dy*dvy
        if(scalar<0):
            d2 = dx*dx+dy*dy
            v2 = dvx*dvx+dvy*dvy
            q = d2-RR*RR
            w = scalar*scalar-q*v2
            #print(counter[counter>1])
            dt = q/(-scalar+sqrt(w))
        else:
            return x,y,vx,vy,dt,ncol
    x,y = propagation(x,y,vx,vy,dt,LBOX)
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    dist = sqrt(dx*dx+dy*dy)
    ndx = dx/dist
    ndy = dy/dist
    vx[i],vx[j],vy[i],vy[j],trash,trash2 = collisional_rule(alpha,0,ndx,ndy,dvx,dvy,vx[i],vx[j],vy[i],vy[j],0,0,SIGMA,0,MASS,COLL_TYPE)
    counter[i] += 1
    counter[j] += 1
    ncol = 1
    return x,y,vx,vy,dt,ncol,counter

def collisional_rule(alpha,beta,ndx,ndy,dvx,dvy,vfxi,vfxj,vfyi,vfyj,wfi,wfj,SIGMA,KAPPA,MASS,model):
    if(model == 1): #TYPE 1: ELASTIC HARD SPHERES
        factor = dvx*ndx+dvy*ndy
        vffxi = vfxi-factor*ndx
        vffxj = vfxj+factor*ndx
        vffyi = vfyi-factor*ndy
        vffyj = vfyj+factor*ndy
    elif(model == 2): #TYPE 2: INELASTIC & SMOOTH HARD SPHERES
        factor = 0.5*(1.+alpha)*(dvx*ndx+dvy*ndy)
        vffxi = vfxi-factor*ndx
        vffxj = vfxj+factor*ndx
        vffyi = vfyi-factor*ndy
        vffyj = vfyj+factor*ndy
    elif(model == 3): #TYPE 3: INELASTIC & ROUGH HARD SPHERES
        beta_red = KAPPA*(1+beta)/(1+KAPPA)
        Sij = SIGMA*0.5*(wfi+wfj)
        factor1 = 0.5*(1.+alpha)*(dvx*ndx+dvy*ndy)
        factor2x = 0.5*beta_red*(dvx*ndy-dvy*ndx-Sij)*ndy
        factor2y = -0.5*beta_red*(dvx*ndy-dvy*ndx-Sij)*ndx
        factor3 = beta_red*(ndx*dvy-ndy*dvx+Sij)/(SIGMA*KAPPA)
        vffxi = vfxi-(factor1*ndx+factor2x)
        vffxj = vfxj+(factor1*ndx+factor2x)
        vffyi = vfyi-(factor1*ndy+factor2y)
        vffyj = vfyj+(factor1*ndy+factor2y)
        wfi = wfi - factor3
        wfj = wfj - factor3
    #FOR OTHER MODELS ADD COLLISIONAL RULES
    return vffxi,vffxj,vffyi,vffyj,wfi,wfj

def collision_langevin(i,j,alpha,tcoll,dt,x,y,vxi,vxj,vyi,vyj,xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj,vfx,vfy,counteri,counterj,LHALF,LBOX,LCELLS,NULL,NULL2,SIGMA,MASS,COLL_TYPE):
    dvx = vfx[i]-vfx[j]
    dvy = vfy[i]-vfy[j]
    dx = x[i]-x[j]
    dy = y[i]-y[j]
    #print('ENTRA EN COLISION')
    ncol = 0
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    scalar = dx*dvx+dy*dvy
    dxx = dx+tcoll*dvx
    dyy = dy+tcoll*dvy
    d2 = dxx*dxx+dyy*dyy
    RR = SIGMA
    if(abs(d2-RR*RR)>NULL2):
        print(tcoll,np.sqrt(d2-RR*RR),LCELLS,dx,dy,dxx,dyy,scalar)
        return x,y,vxi,vxj,vyi,vyj,vfx,vfy,tcoll,ncol,counteri,counterj
    if(abs(d2-RR*RR)>NULL):
        scalar  = dx*dvx+dy*dvy
        if(scalar<0):
            d2 = dx*dx+dy*dy
            v2 = dvx*dvx+dvy*dvy
            q = d2-RR*RR
            w = scalar*scalar-q*v2
            #print(counter[counter>1])
            tcoll = q/(-scalar+sqrt(w))
        else:
            return x,y,vxi,vxj,vyi,vyj,vfx,vfy,tcoll,ncol,counteri,counterj
    x,y = propagation(x,y,vfx,vfy,tcoll,LBOX)
    dx += dvx*tcoll
    dy += dvy*tcoll
    dist = sqrt(dx*dx+dy*dy)
    ndx = dx/dist
    ndy = dy/dist
    vfx[i],vfx[j],vfy[i],vfy[j],trash,trash2 = collisional_rule(alpha,0,ndx,ndy,dvx,dvy,vfx[i],vfx[j],vfy[i],vfy[j],0,0,SIGMA,0,MASS,COLL_TYPE)
    dvx = vxpi-vxpj
    dvy = vypi-vypj
    vxi,vxj,vyi,vyj,trash,trash2 = collisional_rule(alpha,0,ndx,ndy,dvx,dvy,vxpi,vxpj,vypi,vypj,0,0,SIGMA,0,MASS,COLL_TYPE)
    counteri += 1
    counterj += 1
    ncol = 1
    return x,y,vxi,vxj,vyi,vyj,vfx,vfy,tcoll,ncol,counteri,counterj

def collision_langevin_rot(i,j,alpha,beta,tcoll,dt,x,y,vxi,vxj,vyi,vyj,wi,wj,xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj,vfx,vfy,wf,counteri,counterj,LHALF,LBOX,LCELLS,NULL,NULL2,SIGMA,KAPPA,MASS,COLL_TYPE):
    dvx = vfx[i]-vfx[j]
    dvy = vfy[i]-vfy[j]
    dx = x[i]-x[j]
    dy = y[i]-y[j]
    #print('ENTRA EN COLISION')
    ncol = 0
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    #distancia = np.sqrt(dx**2+dy**2)
    scalar = dx*dvx+dy*dvy
    dxx = dx+tcoll*dvx
    dyy = dy+tcoll*dvy
    d2 = dxx*dxx+dyy*dyy
    RR = SIGMA
    if(abs(d2-RR*RR)>NULL2):
        print(tcoll,np.sqrt(d2-RR*RR),LCELLS,dx,dy,dxx,dyy,scalar)
        return x,y,vxi,vxj,vyi,vyj,wi,wj,vfx,vfy,wf,tcoll,ncol,counteri,counterj
    if(abs(d2-RR*RR)>NULL):
        scalar  = dx*dvx+dy*dvy
        if(scalar<0):
            d2 = dx*dx+dy*dy
            v2 = dvx*dvx+dvy*dvy
            q = d2-RR*RR
            wh = scalar*scalar-q*v2
            #print(counter[counter>1])
            tcoll = q/(-scalar+sqrt(wh))
        else:
            return x,y,vxi,vxj,vyi,vyj,wi,wj,vfx,vfy,wf,tcoll,ncol,counteri,counterj
    x,y = propagation(x,y,vfx,vfy,tcoll,LBOX)
    dx += dvx*tcoll
    dy += dvy*tcoll
    dist = sqrt(dx*dx+dy*dy)
    ndx = dx/dist
    ndy = dy/dist
    vfx[i],vfx[j],vfy[i],vfy[j],wf[i],wf[j] = collisional_rule(alpha,beta,ndx,ndy,dvx,dvy,vfx[i],vfx[j],vfy[i],vfy[j],wf[i],wf[j],SIGMA,KAPPA,MASS,COLL_TYPE)
    dvx = vxpi-vxpj
    dvy = vypi-vypj
    vxi,vxj,vyi,vyj,wi,wj = collisional_rule(alpha,beta,ndx,ndy,dvx,dvy,vxpi,vxpj,vypi,vypj,wi,wj,SIGMA,KAPPA,MASS,COLL_TYPE)
    counteri += 1
    counterj += 1
    ncol = 1
    #print('ENTRA EN COLISION')
    return x,y,vxi,vxj,vyi,vyj,wi,wj,vfx,vfy,wf,tcoll,ncol,counteri,counterj

def main_AGF(x,y,vx,vy,dd,f_pointer,cells,counter,counter_t,DT,NSTEP,ALPHA,NPRINT,NPART,NULL,NULL2,DIMENSION,GAMMA,ZETA0,TBATH,MASS,NNCELLS,LCELLS,LBOX,LHALF,SIGMA,NOISE_TYPE,COLL_TYPE):
    time = 0
    ncol = 0
    all_indices = np.arange(len(vx))
    while(time<=NSTEP):
        #PRINT RESULTS
        if(int(time/DT)%10== 0):
            #print(ncol)
            print_results(time,ncol/NPART,x,y,vx,vy,0,f_pointer,MASS,0,NPART,DT,TBATH,DIMENSION,0,1,0,ALPHA,-1,0)
        #PUTATIVE QUANTITIES
        xp,yp,vxp,vyp,espureo = langevin_propagation(x,y,vx,vy,0,DT,NPART,DIMENSION,GAMMA,ZETA0,0,TBATH,MASS,0,NOISE_TYPE,LBOX)
        #FICTIVE VELOCITIES
        vfx = (xp-x)/DT
        vfy = (yp-y)/DT
        #Intialization of Events queue
        Events = []
        pq.heapify(Events)
        cols = 0
        Events = initial_queue(Events,time,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
        index_comp=-1
        list_indices = []
        list_col = []
        time_aux = time
        deltat = 0
        event = 0
        while(deltat>=0 and deltat<= DT):
            #First event is obtained
            if(any(x<0) or any(y<0) or any(x>LBOX) or any(y>LBOX)):
                print('ERROOOOOOOOOOOOOR',event)
            event = pq.heappop(Events)
            #print(time,time_aux,time+DT,event[0])
            if(event[0]>time+DT):
                break
            #We compute the DT_ev
            if(deltat<0):
                print('ERROOOOOOOOOOOOOR',event)
            deltat = event[0]-time
            ddt = event[0] - time_aux
            #If DT_ev \in [0,DT] then, the event occurs
            index1 = event[1][0]
            index2 = event[1][1]
            if(not validity_2(event,time,counter,counter_t)):
                if(index_comp!=index1):
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    index_comp = index1
            else:
                index_comp = -1
                if(index2>=0):
                    vxi,vxj,vyi,vyj = vx[index1],vx[index2],vy[index1],vy[index2]
                    xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj = xp[index1],xp[index2],yp[index1],yp[index2],vxp[index1],vxp[index2],vyp[index1],vyp[index2]
                    x,y,vxp[index1],vxp[index2],vyp[index1],vyp[index2],trash,trash,vfx,vfy,trash,ddt,nc,counter[index1],counter[index2] = collision_langevin_rot(index1,index2,ALPHA,0,ddt,DT,x,y,vxi,vxj,vyi,vyj,0,0,xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj,vfx,vfy,0,counter[index1],counter[index2],LHALF,LBOX,LCELLS,NULL,NULL2,SIGMA,0,MASS,COLL_TYPE)
                    if(nc==0):
                        Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                        continue
                    else:
                        list_indices.append(index1)
                        list_indices.append(index2)
                        list_col.append(index1)
                        list_col.append(index2)
                        time_aux += ddt
                        #print(ncol)
                    ncol += nc
                    cols += nc
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    Events = res_events(Events,index2,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                elif(index2<0):
                    time_aux += ddt
                    list_indices.append(index1)
                    x,y,counter_t[index1],dd,cells = transfer(index1,index2,ddt,NULL,x,y,vfx,vfy,dd,cells,counter_t[index1],NNCELLS,LCELLS,LBOX)
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
        #If no collision occurs
        time += DT
        range_indices = np.delete(all_indices,list_indices)
        vx,vy = vxp,vyp
        x[list_indices],y[list_indices] = propagation(x[list_indices],y[list_indices],vfx[list_indices],vfy[list_indices],DT-deltat,LBOX)
        x[range_indices],y[range_indices] = xp[range_indices],yp[range_indices]

def main_AGF_rot(x,y,vx,vy,w,dd,f_pointer,dire,cells,counter,counter_t,DT,NSTEP,ALPHA,BETA,NPRINT,NPART,NULL,NULL2,DIMENSION_T,DIMENSION_R,GAMMA,ZETA0,EPSILON,TBATH,MASS,INERTIA,NNCELLS,LCELLS,LBOX,LHALF,SIGMA,KAPPA,NOISE_TYPE,COLL_TYPE):
    time = 0
    ncol = 0
    all_indices = np.arange(len(vx))
    scale = 1
    while(time<=NSTEP):
        #PRINT RESULTS
        if(int(time/DT)%250== 0):
            #print(ncol)
            kld = KLD_densityfluctuations(x,y,cells,NPART,LBOX)
            print_results(time,ncol/float(NPART),x,y,vx,vy,w,f_pointer,dire,MASS,INERTIA,NPART,DT,TBATH,DIMENSION_T,DIMENSION_R,scale,kld,ALPHA,BETA,0)
        if(NOISE_TYPE == 0 and ncol%int(2*NPART)==0):
            vx,vy,w,isca = rescale_rot(vx,vy,w,MASS,INERTIA)
            scale /= isca
           # print(scale)
        #PUTATIVE QUANTITIES
        xp,yp,vxp,vyp,w = langevin_propagation(x,y,vx,vy,w,DT,NPART,DIMENSION_T,GAMMA,ZETA0,EPSILON,TBATH,MASS,INERTIA,NOISE_TYPE,LBOX)
        #FICTIVE VELOCITIES
        vfx = (xp-x)/DT
        vfy = (yp-y)/DT
        wf = 1.*w
        wp = 1.*w
        #Intialization of Events queue
        Events = []
        pq.heapify(Events)
        cols = 0
        Events = initial_queue(Events,time,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
        index_comp=-1
        list_indices = []
        list_col = []
        time_aux = time
        deltat = 0
        #print(Events)
        while(deltat>=0 and deltat<= DT):
            #First event is obtained
            event = pq.heappop(Events)
            #print(time,time_aux,time+DT,event[0])
            if(event[0]>time+DT):
                break
            #We compute the DT_ev
            if(deltat<0):
                print('ERROOOOOOOOOOOOOR',event)
            deltat = event[0]-time
            ddt = event[0] - time_aux
            #If DT_ev \in [0,DT] then, the event occurs
            index1 = event[1][0]
            index2 = event[1][1]
            valid, time_caca = validity(event,time,counter,counter_t)
            if(not valid):
                if(index_comp!=index1):
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    index_comp = index1
            else:
                #print(event)
                index_comp = -1
                if(index2>=0):
                    vxi,vxj,vyi,vyj,wi,wj = vx[index1],vx[index2],vy[index1],vy[index2],w[index1],w[index2]
                    xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj = xp[index1],xp[index2],yp[index1],yp[index2],vxp[index1],vxp[index2],vyp[index1],vyp[index2]
                    x,y,vxp[index1],vxp[index2],vyp[index1],vyp[index2],wp[index1],wp[index2],vfx,vfy,wf,ddt,nc,counter[index1],counter[index2] = collision_langevin_rot(index1,index2,ALPHA,BETA,ddt,DT,x,y,vxi,vxj,vyi,vyj,wi,wj,xpi,xpj,ypi,ypj,vxpi,vxpj,vypi,vypj,vfx,vfy,wf,counter[index1],counter[index2],LHALF,LBOX,LCELLS,NULL,NULL2,SIGMA,KAPPA,MASS,COLL_TYPE)
                    if(nc==0):
                        Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                        continue
                    else:
                        list_indices.append(index1)
                        list_indices.append(index2)
                        list_col.append(index1)
                        list_col.append(index2)
                        time_aux += ddt
                        #print(ncol)
                    ncol += nc
                    cols += nc
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    Events = res_events(Events,index2,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                elif(index2<0):
                    time_aux += ddt
                    list_indices.append(index1)
                    x,y,counter_t[index1],dd,cells = transfer(index1,index2,ddt,NULL,x,y,vfx,vfy,dd,cells,counter_t[index1],NNCELLS,LCELLS,LBOX)
                    Events = res_events(Events,index1,time_aux,dd,x,y,vfx,vfy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
        #If no collision occurs
        time += DT
        range_indices = np.delete(all_indices,list_indices)
        vx,vy = vxp,vyp
        w = wp
        x[list_indices],y[list_indices] = propagation(x[list_indices],y[list_indices],vfx[list_indices],vfy[list_indices],DT-deltat,LBOX)
        x[range_indices],y[range_indices] = xp[range_indices],yp[range_indices]
    return x,y,vx,vy,w,time_aux,cols,dd,cells,counter,counter_t
        
def EDMD_rot(x,y,vx,vy,w,dd,f_pointer,dire,cells,counter,counter_t,cont_over,time,collisions_0,NSTEP,NRESCALE,alpha,beta,NPRINT,NPART,NULL,NULL2,DIMENSION_T,DIMENSION_R,MASS,INERTIA,NNCELLS,LCELLS,LBOX,LHALF,SIGMA,KAPPA,COLL_TYPE):
    Events = []
    pq.heapify(Events)
    cols = collisions_0
    Events = initial_queue(Events,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
    index_comp=-1
    #index_comp2 = -1
    event = 0
    time_aux = time
    scale = 1.
    indices_all = np.array([i for i in range(len(x))])
    kld = KLD_densityfluctuations(x,y,cells,NPART,LBOX)
    print_results(time,cols/NPART,x,y,vx,vy,w,f_pointer,dire,MASS,INERTIA,NPART,1,1,DIMENSION_T,DIMENSION_R,scale,kld,alpha,beta,0)
    #print('Se hace copia')
    x_copy = np.copy(x)
    y_copy = np.copy(y)
    vx_copy = np.copy(vx)
    vy_copy = np.copy(vy)
    w_copy = np.copy(w)
    t_copy = time
    dd_copy = copy.deepcopy(dd)
    counter_t_copy = copy.deepcopy(counter_t)
    counter_copy = copy.deepcopy(counter)
    cells_copy = copy.deepcopy(cells)
    cols_copy = collisions_0
    #flag_check = False
    #NOVERLAP = NRESCALE
    ff = False
    fl_col = False
    cont_write = 0
    cont_write_copy = cont_write
    cont_over = 0
    index1_past = -1
    index2_past = -1
    while(cols< NSTEP*NPART+collisions_0):
        cont_over +=1
        #if(any(x<0) or any(y<0) or any(x>LBOX) or any(y>LBOX)):
            #ff = True
            #inddd = indices_all[(x<0)^(x>LBOX)^(y<0)^(y>LBOX)]
            #print('ERROOOR',inddd,vx[inddd],vy[inddd],w[inddd])
            #plot_density(np.copy(x),np.copy(y),np.copy(vx),np.copy(vy),LBOX)
            #x,y,dd,cells = restore_pos(x,y,dd,cells,NULL,LCELLS,LBOX)
            #Events = []
            #pq.heapify(Events)
            #Events = initial_queue(Events,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
        event = pq.heappop(Events)
        #print(event)
        #if(flag_check):
        #    print(event)
        #    flag_check = False
        index1 = event[1][0]
        index2 = event[1][1]
        time_prime = event[0]
        dt = time_prime-time_aux
        valid,time_prime = validity(event,time_aux,counter,counter_t)
        if(not valid):
            if(dt>=0):
                Events = res_events(Events,index1,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                #index_comp = index1
            #else:
                #print(event)
                #flag_check = False
            #elif(contador == 1):
                #Events = []
                #pq.heapify(Events)
                #Events = initial_queue(Events,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
                #contador+=1
            #else:
                #
                #index_comp = -1
        else:
            if(dt==0):
                dt = 0
                #continue
                #ff=True
                #break
                #print('PASA',flush=True)
            index_comp = -1
            #print(event,cols/NPART, flush=True)
            if(index2>=0):
                if(index1==index1_past and index2==index2_past):
                    #print('Colision repetida')
                    TIMEC = 1/((SIGMA/LBOX)**2*NPART*np.sqrt(np.average(vx**2+vy**2)))
                    x,y,vx,vy,w,dt,nc,counter,ff = collision_rot(index1,index2,1,-1,dt,x,y,vx,vy,w,counter,LHALF,LBOX,LCELLS,SIGMA,KAPPA,TIMEC,NULL,NULL2,MASS,COLL_TYPE,fl_col)
                else:
                    TIMEC = 1/((SIGMA/LBOX)**2*NPART*np.sqrt(np.average(vx**2+vy**2)))
                    x,y,vx,vy,w,dt,nc,counter,ff = collision_rot(index1,index2,alpha,beta,dt,x,y,vx,vy,w,counter,LHALF,LBOX,LCELLS,SIGMA,KAPPA,TIMEC,NULL,NULL2,MASS,COLL_TYPE,fl_col)
                    index1_past = index1
                    index2_past = index2
                if(nc==0):
                    Events = res_events(Events,index1,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    continue
                else:
                    time_aux += dt
                    cols += nc
                    #print(cols)
                    if(nc==1 and cols%NRESCALE == 0):
                        vx,vy,w,isca = rescale_rot(vx,vy,w,MASS,INERTIA)
                        scale /= isca
                        #density_fluctuations(x,y,NPART,LCELLS,LBOX,NNCELLS)
                        #plot_density(np.copy(x),np.copy(y),np.copy(vx),np.copy(vy),LBOX,NPART)
                        #Energy_fluctuations(vx,vy,w,MASS,INERTIA)
                        if(abs(np.average(vx))>NULL2 or abs(np.average(vy))>NULL2):
                            vx -= np.average(vx)
                            vy -= np.average(vy)
                            #w -= np.average(w)
                            #print('ESTA PASANDO')
                        Events = []
                        pq.heapify(Events)
                        Events = initial_queue(Events,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
                    else:
                        Events = res_events(Events,index1,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                        Events = res_events(Events,index2,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                    if(nc == 1 and cols%NPRINT== 0):
                        #print(ncol)
                        kld = KLD_densityfluctuations(x,y,cells,NPART,LBOX)
                        print_results(time_aux,cols/NPART,x,y,vx,vy,w,f_pointer,dire,MASS,INERTIA,NPART,1,1,DIMENSION_T,DIMENSION_R,scale,kld,alpha,beta,cont_write)
                        cont_write += 1
                    #if(nc == 1 and cols%(10*NPART)== 0):
                        #plot_density(np.copy(x),np.copy(y),np.copy(vx),np.copy(vy),LBOX,NPART)
                fl_col = True
            elif(index2<0):
                fl_col = False
                time_aux += dt
                x,y,counter_t[index1],dd,cells = transfer(index1,index2,dt,NULL,x,y,vx,vy,dd,cells,counter_t[index1],NNCELLS,LCELLS,LBOX)
                Events = res_events(Events,index1,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
                #if(any(x<-NULL) or any(y<-NULL) or any(x>LBOX+NULL) or any(y>LBOX+NULL)):
                #    x,y,dd,cells = restore_pos(x,y,dd,cells,NULL,LCELLS,LBOX)
                #    Events = []
                #    pq.heapify(Events)
                #   Events = initial_queue(Events,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
        #fl_out = (any(x<0)^any(y<0)^any(x>LBOX)^any(y>LBOX))
        #for i in range(NPART):
        #    if(check_overlap(x,y,i,dd,cells,NNCELLS,LBOX,SIGMA,NULL)):
        #            ff = True
        if(any(x<0) or any(y<0) or any(x>LBOX) or any(y>LBOX)):
            ff = True
        if(cont_over%int(2*NPART*0.01) == 0 or ff):
            #cont_over = 0
            flag_overlap = False
            for i in range(NPART):
                if(check_overlap(x,y,i,dd,cells,NNCELLS,LBOX,SIGMA,NULL)):
                    flag_overlap = True
            if(flag_overlap or ff):
                #flag_check =True
                time_aux = t_copy
                cols = cols_copy
                x = np.copy(x_copy)
                y = np.copy(y_copy)
                vx = np.copy(vx_copy)
                vy = np.copy(vy_copy)
                w = np.copy(w_copy)
                dd = copy.deepcopy(dd_copy)
                counter_t = copy.deepcopy(counter_t_copy)
                counter = copy.deepcopy(counter_copy)
                cells = copy.deepcopy(cells_copy)
                cont_write = cont_write_copy
                #plot_density(np.copy(x),np.copy(y),np.copy(vx),np.copy(vy),LBOX,NPART)
                Events = []
                pq.heapify(Events)
                Events = initial_queue(Events,time_aux,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART)
               # print('DONE')
            else:
                t_copy = time_aux
                cols_copy = cols
                x_copy = np.copy(x)
                y_copy = np.copy(y)
                vx_copy = np.copy(vx)
                vy_copy = np.copy(vy)
                w_copy = np.copy(w)
                dd_copy = copy.deepcopy(dd)
                counter_t_copy = copy.deepcopy(counter_t)
                counter_copy = copy.deepcopy(counter)
                cells_copy = copy.deepcopy(cells)
                cont_write_copy = cont_write
            ff = False
      #  elif(cont_over%int(NPART*0.01) == 0):
            #print('Se hace copia')
      #      t_copy = time_aux
      #      cols_copy = cols
      #      x_copy = np.copy(x)
      #      y_copy = np.copy(y)
      #      vx_copy = np.copy(vx)
      #      vy_copy = np.copy(vy)
      #      w_copy = np.copy(w)
      #      dd_copy = copy.deepcopy(dd)
      #      counter_t_copy = copy.deepcopy(counter_t)
      #      counter_copy = copy.deepcopy(counter)
      #      cells_copy = copy.deepcopy(cells)
      #      cont_write_copy = cont_write 
    return x,y,vx,vy,w,time_aux,cols,dd,cells,counter,counter_t,cont_over

def restore_pos(x,y,dd,cells,NULL,LCELLS,LBOX):
    xmen = x<-NULL
    ymen = y<-NULL
    xplus = x>LBOX+NULL
    yplus = y>LBOX+NULL
    x[x<-NULL] += LBOX
    y[y<-NULL] += LBOX
    x[x>LBOX+NULL] -= LBOX
    y[y>LBOX+NULL] -= LBOX
    indices = np.array([i for i in range(len(x))])
    indices_mod = indices[xmen^ymen^xplus^yplus]
    for i in indices_mod:
        cellx = int(x[i]//LCELLS)
        celly = int(y[i]//LCELLS)
        cellx_old,celly_old =dd[i]
        cells[cellx_old][celly_old].remove(i)
        cells[cellx][celly].append(i)
        dd[i] = (cellx,celly)
    return x,y,dd,cells
def plot_density(x,y,vx,vy,LBOX,NPART):
    from matplotlib.colors import LinearSegmentedColormap
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
    ], N=256)
    def using_mpl_scatter_density(fig, x, y,LBOX):
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.set_xlim(0,LBOX)
        ax.set_ylim(0,LBOX)
        density = ax.scatter_density(x, y, cmap=white_viridis)
        fig.colorbar(density, label='Number of points per pixel')
    fig = plt.figure()
    using_mpl_scatter_density(fig, x, y,LBOX)
    randd = np.random.uniform(0,100)
    #plt.savefig('/home/alberto/DOCTORADO/SEGUNDO/Transport_coefficients/prueba-error13/5/cluster_'+str(randd)+'.png')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.set_xlim(0,LBOX)
    ax.set_ylim(0,LBOX)
    ax.quiver(x,y,vx,vy)
    #plt.savefig('/home/alberto/DOCTORADO/SEGUNDO/Transport_coefficients/prueba-error13/5/vector_'+str(randd)+'.png')
    plt.show()

def print_results(time,col,x,y,vx,vy,w,f_pointer,dire,MASS,INERTIA,NPART,DT,TBATH,DIMENSION_T,DIMENSION_R,scale,KLD,ALPHA,BETA,cont):
    temp = Temperature(vx,vy,MASS)
    if(DIMENSION_R != 0):
        temp_r = Temp_rot(w,INERTIA)
        a02 = a02_cumulant(w,temp_r,INERTIA,DIMENSION_R)
        a11 = a11_cumulant(vx,vy,w,temp,temp_r,MASS,INERTIA,DIMENSION_T,DIMENSION_R)
        total_T = (2*scale*temp+scale*temp_r)/3.
    else:
        temp_r = 1
        a02 = 0
        a11 = 0
    a20 = a20_cumulant(vx,vy,temp,MASS,DIMENSION_T)
    #print(temp)
    #if(int(time/DT)%1 == 0):
        #print('time:',(time/np.sqrt(scale)))
        #print('collisions per particle:',col)
    #    if(DIMENSION_R!=0):
    #        print('Theta:',temp_r/temp,'\n')
    #        print('Translational Temperature:', scale*temp)
    #        print('Rotational Temperature:', scale*temp_r)
    #        print('Total temperature:', (2*scale*temp+scale*temp_r)/3.)
    #    else:
    #        print('Total temperature:', (2*scale*temp+scale*temp_r)/3.)
    if(DIMENSION_R != 0):
        f_pointer.write(str(time/np.sqrt(scale))+'\t'+str(col)+'\t'+str(total_T)+'\t'+str(temp_r/temp)+'\t'+str(a20)+'\t'+str(a02)+'\t'+str(a11)+'\t'+str(np.average(vx))+'\t'+str(np.average(vy))+'\t'+str(np.average(w))+'\t'+str(KLD)+'\n')
    else:
        f_pointer.write(str(time/np.sqrt(scale))+'\t'+str(col)+'\t'+str(scale*temp/temp_r)+'\t'+str(a20)+'\n')
    direc = dire
    #cont = int(time/(250*DT))
    file_write_posvels(direc,cont,x,y,0,vx,vy,0,0,0,w,2)

def check_overlap(x,y,a,dd,cells,NNCELLS,LBOX,SIGMA,NULL):
    cella = dd[a]
    cellx = cella[0]
    celly = cella[1]
    flag_over = False
    if(celly == NNCELLS-1):
        celly_p1 = 0
    else:
        celly_p1 = celly + 1
    if(celly == 0):
        celly_m1 = NNCELLS-1
    else:
        celly_m1 = celly - 1
    if(cellx == NNCELLS-1):
        cellx_p1 = 0
    else:
        cellx_p1 = cellx + 1
    if(cellx == 0):
        cellx_m1 = NNCELLS-1
    else:
        cellx_m1 = cellx - 1
    #SIGMA = SIGMA + NULL
    if(a not in cells[cellx][celly]):
        print('Algo pasa')
    for i in cells[cellx][celly]:
        if(i != a):
            dx = fmod(x[a]-x[i],LBOX)
            dy = fmod(y[a]-y[i],LBOX)
            d2 = dx*dx+dy*dy
            if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
                #print('overlap',a,i,d2-SIGMA*SIGMA)
                flag_over = True
    for i in cells[cellx_m1][celly]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA  and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx_p1][celly]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx][celly_m1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx][celly_p1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx_m1][celly_m1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx_m1][celly_p1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx_p1][celly_m1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    for i in cells[cellx_p1][celly_p1]:
        dx = fmod(x[a]-x[i],LBOX)
        dy = fmod(y[a]-y[i],LBOX)
        d2 = dx*dx+dy*dy
        if(d2<SIGMA*SIGMA and abs(d2-SIGMA*SIGMA)>NULL):
            #print('overlap',a,i,d2-SIGMA*SIGMA)
            flag_over = True
    return flag_over
    


def transfer(i,w,dt,NULL,x,y,vx,vy,dd,cells,counter_ti,NNCELLS,LCELLS,LBOX):
    cell = dd[i]
    cx = cell[0]
    cy = cell[1]
    x,y = propagation(x,y,vx,vy,dt,LBOX)
    if i not in cells[cx][cy]:
            print(i,cells[cx][cy])
            print('ERROR')
            return x,y 
    else:
        cells[cx][cy].remove(i)
        if(cx == NNCELLS-1):
            cx_p1 = 0
        else:
            cx_p1 = cx+1
        if(cx == 0):
            cx_m1 = NNCELLS-1
        else:
            cx_m1 = cx-1
        if(cy == NNCELLS-1):
            cy_p1 = 0
        else:
            cy_p1 = cy+1
        if(cy == 0):
            cy_m1 = NNCELLS-1
        else:
            cy_m1 = cy-1
        if(w == -1):
            if(vx[i]<0):
                cells[cx_m1][cy].append(i)
                dd[i] = (cx_m1,cy)
                if(cx == 0):
                    x[i] = LBOX-NULL
                else:
                    x[i] = cx*LCELLS-NULL
            elif(vx[i]>0):
                cells[cx_p1][cy].append(i)
                dd[i] = (cx_p1,cy)
                if(cx == NNCELLS-1):
                    x[i] = 0.+NULL
                else:
                    x[i] = cx_p1*LCELLS+NULL
        elif(w==-2):
            if(vy[i]<0):
                cells[cx][cy_m1].append(i)
                dd[i] = (cx,cy_m1)
                if(cy == 0):
                    y[i] = LBOX-NULL
                else:
                    y[i] = cy*LCELLS-NULL
            elif(vy[i]>0):
                cells[cx][cy_p1].append(i)
                dd[i] = (cx,cy_p1)
                if(cy == NNCELLS-1):
                    y[i] = 0.+NULL
                else:
                    y[i] = cy_p1*LCELLS+NULL
        counter_ti += 1
    return x,y,counter_ti,dd,cells
            
def ppcoll(i,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX):
    #print('ppcoll',j)
    dx = x[i]-x[j]
    dy = y[i]-y[j]
    dvx = vx[i]-vx[j]
    dvy = vy[i]-vy[j]
    celli = dd[i]
    cellj = dd[j]
    cellxi = celli[0]
    cellxj = cellj[0]
    cellyi = celli[1]
    cellyj = cellj[1]
    dx,dy = boundary(dx,dy,LHALF,LBOX)
    scalar = dx*dvx+dy*dvy
    if(scalar >= 0):
        return np.Infinity
    else:
        d2 = dx*dx+dy*dy
        if(d2-SIGMA*SIGMA<=0):
            return time
        else:
            RR = SIGMA+1.e-20
        v2 = dvx*dvx+dvy*dvy
        q = d2-RR*RR
        w = scalar*scalar-q*v2
        if(w<0):
            return np.Infinity
        else:
            ct = time+q/(-scalar+sqrt(w))
            xci = x[i]+(ct-time)*vx[i]
            xcj = x[j]+(ct-time)*vx[j]
            if((xci>((cellxi+1)*LCELLS))or(xci<((cellxi)*LCELLS))or(xcj>((cellxj+1)*LCELLS))or(xcj<((cellxj)*LCELLS))):
                return np.Infinity
            else:
                yci = y[i]+(ct-time)*vy[i]
                ycj = y[j]+(ct-time)*vy[j]
                if((yci>((cellyi+1)*LCELLS))or(yci<(cellyi*LCELLS))or(ycj>((cellyj+1)*LCELLS))or(ycj<(cellyj*LCELLS))):
                    return np.Infinity
                else:
                    return ct

def pwcoll(i,w,time,cx,cy,dd,x,y,vx,vy,LCELLS):
    LCELLS = LCELLS + 1.e-20
    if(w==-1): #horizontal wall. At first, only 1-cell
        if(vx[i]>0): #transfer with the Right wall.
            ct = time + ((cx+1)*LCELLS-x[i])/vx[i]
            return ct
        elif(vx[i]<0): #transfer with the Left wall
            ct = time + (cx*LCELLS-x[i])/vx[i]
            return ct
        else:
            return np.Infinity
    elif(w==-2): #vertical wall.
        if(vy[i]>0): #collision with the top wall.
            ct = time + ((cy+1)*LCELLS-y[i])/vy[i]
            return ct
        elif(vy[i]<0): #collision with the bottom wall
            ct = time + (cy*LCELLS-y[i])/vy[i]
            return ct
        else:
            return np.Infinity
    else:
        print('ERROR')
        return np.Infinity

def validity(event,tt,counter,counter_t):
    time = event[0]
    pair = event[1]
    cc = event[2]
    cc_t = event[3]
    if(time>=0 and time != np.Infinity and ((cc[1]<0 and cc[0] == counter[pair[0]] and cc_t[0] == counter_t[pair[0]])or(cc[0] == counter[pair[0]] and cc[1] == counter[pair[1]] and cc_t[0] == counter_t[pair[0]] and cc_t[1] == counter_t[pair[1]])) and time>=tt):
        if(time==tt):
            #print(time)
            if(pair[1]>=0):
                return True,time
            else:    
                return True,time+1.e-15
        else:
            return True,time
    else:
        return False,time

def validity_2(event,tt,counter,counter_t):
    time = event[0]
    pair = event[1]
    cc = event[2]
    cc_t = event[3]
    if(time>=0 and time != np.Infinity and ((cc[1]<0 and cc[0] == counter[pair[0]] and cc_t[0] == counter_t[pair[0]])or(cc[0] == counter[pair[0]] and cc[1] == counter[pair[1]] and cc_t[0] == counter_t[pair[0]] and cc_t[1] == counter_t[pair[1]])) and time>=tt):
        return True
    else:
        return False

def predict(a,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX):
    cell = dd[a]
    cellx = cell[0]
    celly = cell[1]
    coll_list = []
    pq.heapify(coll_list)
    if a not in cells[cellx][celly]:
        #print(a,cells[cellx][celly],cellx,celly)
        #print(cells)
        print('ERROR en predict')
    else:
        for j in cells[cellx][celly]:
            if(a!=j):
                #print('predict',j,cells[cellx][celly])
                time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
                if(time_prime!=np.Infinity):
                    pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        if(NNCELLS==1):
            if(coll_list!=[]):
                popped = pq.heappop(coll_list)
                return popped
            else:
                return np.Infinity
        if(celly == NNCELLS-1):
            celly_p1 = 0
        else:
            celly_p1 = celly + 1
        if(celly == 0):
            celly_m1 = NNCELLS-1
        else:
            celly_m1 = celly - 1
        if(cellx == NNCELLS-1):
            cellx_p1 = 0
        else:
            cellx_p1 = cellx + 1
        if(cellx == 0):
            cellx_m1 = NNCELLS-1
        else:
            cellx_m1 = cellx - 1
        for j in cells[cellx][celly_p1]:
            #print('predict 2',j,cells[cellx][celly_p1])
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_p1][celly]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_p1][celly_p1]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx][celly_m1]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_m1][celly]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_m1][celly_m1]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_m1][celly_p1]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
        for j in cells[cellx_p1][celly_m1]:
            time_prime = ppcoll(a,j,time,dd,x,y,vx,vy,SIGMA,LCELLS,LHALF,LBOX)
            if(time_prime!=np.Infinity):
                pq.heappush(coll_list,(time_prime,(a,j),(counter[a],counter[j]),(counter_t[a],counter_t[j])))
    w = -1 #Horizontal transfer
    time_prime = pwcoll(a,w,time,cellx,celly,dd,x,y,vx,vy,LCELLS)
    pq.heappush(coll_list,(time_prime,(a,w),(counter[a],-1),(counter_t[a],-1)))
    w = -2 #Vertical transfer
    time_prime = pwcoll(a,w,time,cellx,celly,dd,x,y,vx,vy,LCELLS)
    pq.heappush(coll_list,(time_prime,(a,w),(counter[a],-2),(counter_t[a],-2)))
    if(coll_list!=[]):
        popped = pq.heappop(coll_list)
        return popped
    else:
        return np.Infinity

def res_events(Events,a,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX):
    event = predict(a,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
    pq.heappush(Events,event)
    return Events


def initial_queue(Events,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX,NPART):
    for i in range(NPART):
        event = predict(i,time,dd,x,y,vx,vy,cells,counter,counter_t,NNCELLS,SIGMA,LCELLS,LHALF,LBOX)
        if(event!=np.Infinity):
            pq.heappush(Events, event)
    return Events

def freq_nu0(SIGMA,DENSITY,DIMENSION,MASS,TBATH):
    return np.sqrt(2*np.pi**(DENSITY-1))*DENSITY*SIGMA**DIMENSION*np.sqrt(2*TBATH/MASS)/sc.gamma(DIMENSION/2)

def KLD_densityfluctuations(x,y,cells,NPART,LBOX):
    sumaKLD = 0
    Ndiv = 10
    Number_cells = Ndiv**2
    Lmeas = LBOX/Ndiv
    for i in range(1,Ndiv+1):
        for j in range(1,Ndiv+1):
            Nk = len(x[np.logical_and(x<i*Lmeas,np.logical_and(x>=(i-1)*Lmeas,np.logical_and(y<j*Lmeas,y>=(j-1)*Lmeas)))])
            if(Nk>0):
                sumaKLD +=Nk*np.log(Nk*Number_cells/NPART)
    sumaKLD /= NPART
    return sumaKLD

def Energy_fluctuations(vx,vy,w,MASS,INERTIA):
    energy = 0.5*MASS*(vx**2+vy**2)
    energy_mean = np.mean(energy)
    delta_energy = energy/energy_mean-1
    #variance_energy = np.var(energy)
    plt.hist(delta_energy, bins = 100)
    #plt.yscale('log')
    plt.show() 

def density_fluctuations(x,y,NPART,LCELLS,LBOX,NNCELLS):
    Lsub = LCELLS/2.
    pos = np.linspace(Lsub/2,LBOX-Lsub/2,NNCELLS*2-1)
    fluct_nx = [len(x[np.logical_and(x>=i*Lsub,x<(i+1)*Lsub)])*LBOX/(Lsub*NPART)-1 for i in range(NNCELLS*2-1)]
    fluct_ny = [len(y[np.logical_and(y>=i*Lsub,y<(i+1)*Lsub)])*LBOX/(Lsub*NPART)-1 for i in range(NNCELLS*2-1)]
    Nav = 10
    pos_av = [0.5*(pos[i]+pos[i+Nav-1]) for i in range(len(pos)-Nav+1)]
    nx_av = np.convolve(fluct_nx, np.ones(Nav)/Nav, mode='valid')
    ny_av = np.convolve(fluct_ny, np.ones(Nav)/Nav, mode='valid')
    plt.plot(pos,fluct_nx)
    plt.plot(pos_av,nx_av)
    plt.show()

def file_write_posvels(dir,cont,x,y,z,vx,vy,vz,wx,wy,wz,dim):
    if(dim == 2):
        np.savetxt(dir+'_'+str(cont)+'.txt',(x,y,vx,vy,wz))
    else:
        np.savetxt(dir+'_'+str(cont)+'.txt',(x,y,z,vx,vy,vz,wx,wy,wz))



    
