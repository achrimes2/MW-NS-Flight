"""
@author: Ashley Chrimes
Interactive Milky Way fraction of light tool for Galactic neutron star populations
See Chrimes et al. (2021) for details
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
from matplotlib import colors
from matplotlib import cm
from matplotlib.widgets import Slider

# Olausen et al. 2014 McGill catalogue data (http://www.physics.mcgill.ca/~pulsar/magnetar/main.html)
MWmagnetars = pd.read_csv('TabO1.csv')       
distances = MWmagnetars.Dist.values


#-------------------------------------------------------------------------------------------------------------#
############### FRACTIONAL LIGHT #########################################################
#-------------------------------------------------------------------------------------------------------------#
fig = plt.figure(figsize = (12,4))
plt.subplots_adjust(bottom=0.5,top=0.95)
ax1 = fig.add_subplot(121)
p = ax1.plot([-1,-2],[0,1])

# Setting up widgets
ax_slide = plt.axes([0.25, 0.1, 0.55, 0.03]) 
s_factor = Slider(ax_slide, 'R$_\mathrm{selection}$ [kpc]', 0.3*30, 0.5*30, valinit=0.4*30, valstep=0.01*30)

ax_slide2 = plt.axes([0.25, 0.05, 0.55, 0.03])
s_factor2 = Slider(ax_slide2, 'Res. [kpc pix$^{-1}$]', 0.25, 1, valinit=30/120, valstep=0.05)

ax_slide3 = plt.axes([0.25, 0.15, 0.55, 0.03])
s_factor3 = Slider(ax_slide3, 'L$_\mathrm{disc}$ reduction', 1.0, 25, valinit=2.7, valstep=0.1)

ax_slide5 = plt.axes([0.25, 0.2, 0.55, 0.03])
s_factor5 = Slider(ax_slide5, 'L$_\mathrm{arm}$ reduction', 1, 25, valinit=2.7*0.75, valstep=0.1)

ax_slide4 = plt.axes([0.25, 0.25, 0.55, 0.03])
s_factor4 = Slider(ax_slide4, 'L$_\mathrm{bulge}$ reduction', 1.0, 25, valinit=5.6, valstep=0.1)



# Creating the interactive figure with default slider values
maxdistlim = 15 #kpc
ylim = 8.3 #s_factor7.val
ymag = np.loadtxt('mcgill_y.txt')
names = MWmagnetars.Name.values[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
xmag = np.loadtxt('mcgill_x.txt')[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
hmag = np.loadtxt('mcgill_h.txt')[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]   
derrup = MWmagnetars.Dist_EUp.values[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
derrlo = MWmagnetars.Dist_EDn.values[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
dmag = distances[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
ymag = np.loadtxt('mcgill_y.txt')[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]   

# Radius within which to select pixels for Flight, and the image resolution
circlecut = s_factor.val/30
Res = int(30/(s_factor2.val) - 1) 
R = 30/(Res+1)

# Factor by which to reduce the pixel values of the components
disc_redux = s_factor3.val
bulgebar_redux = s_factor4.val
arm_redux = s_factor5.val

# Loading the component images with the chosen resolution and flux scaling 
# These were created with the Urquhart et al. 2014 masers and the method of Reid et al. 2019
# For details of this and the other components, see Chrimes et al. (2021)
arms = np.loadtxt('Fullmap/Arms_'+str(Res)+'.txt')/(arm_redux)
disc_in = np.loadtxt('Fullmap/Disc_'+str(Res)+'.txt')/(disc_redux)
disc = scipy.ndimage.gaussian_filter(disc_in,sigma=5)
barbulge = np.loadtxt('Fullmap/Barbulge_'+str(Res)+'.txt')/(bulgebar_redux)

# The image
Resgrid = scipy.ndimage.gaussian_filter(arms + disc + barbulge,sigma=0.75,truncate=2)

# Creating the image
ax2 = fig.add_subplot(122)
ax2.imshow(Resgrid,cmap=cm.Oranges,norm=colors.LogNorm()) 
ax2.invert_yaxis()
ax2.set_yticks([])
ax2.set_xticks([])

theta = np.linspace(0, 2*np.pi, 100)
rselection = s_factor.val*((Res+1)/30)
XS = rselection*np.cos(theta) + (0/R + Res/2)
YS = rselection*np.sin(theta) + ((8.2-8.2)/R + Res/2)
ax2.plot(XS,YS,color='cadetblue',linestyle='--')

ax2.set_xlim([0,Res+1])
ax2.set_ylim([0,(Res+1)/2])

# Fraction of light calculation begins
shiftedx = (xmag[1:21]/R + Res/2)
shiftedy = ((ymag[1:21]-8.2)/R + Res/2)
ax2.scatter(shiftedx,shiftedy)
ax2.plot((0/R + Res/2),((0-8.2)/R + Res/2),'ok',fillstyle='none',markersize=10)
ax2.plot((0/R + Res/2),((0-8.2)/R + Res/2),'+k',markersize=10)

ax2.invert_yaxis()

# Geting all pixel values and sorting them into increasing order.
zlist = []
rlist = []
indicies = []
II = 0
mask = np.ones((Res+1,Res+1))    
indexes = np.linspace(0,Res,Res+1).astype(int)
for valx in indexes:
    for valy in indexes[0:int((Res+1)/2)]:   
        zlist = zlist + [Resgrid[valy,valx]]   
        rval = np.sqrt((valx-(Res+1)/2)**2 + (valy-(Res+1)/2)**2)
        rlist = rlist + [rval] #radial distance from centre of image
        indicies.append(II)
        if rval < circlecut*(Res+1):  
            mask[valy,valx] = np.nan
        II = II + 1
zlist = np.array(zlist)
rlist = np.array(rlist)
indicies = np.array(indicies)

indicies = indicies[rlist < circlecut*(Res+1)]    # selecting the pixel indicies that satisify circlecut
zlist = zlist[rlist < circlecut*(Res+1)]   
zsort, indexsort = zip(*sorted(zip(zlist,indicies)))    #indicies sorted by z value

zsort = np.array(zsort)
flcumul = []
cflux = 0
for entry in zsort:
    cflux = cflux + entry
    flcumul.append(cflux)
flight_masers = np.array(flcumul)/np.max(flcumul)  #every pixel gets assigned a value from 0 to 1

# Ordering x,y PIXEL coordinates so that each z value has a coordinate pair in the same order as z and the indicies
xlist = []
ylist = []
c = 0
for Xx in indexes:
    for Yy in indexes[0:int((Res+1)/2)]:   
        if c in indicies:  
            xlist.append(Xx)
            ylist.append(Yy)  
        c = c + 1
zsort, xsort = zip(*sorted(zip(zlist, xlist))) 
zsort, ysort = zip(*sorted(zip(zlist, ylist))) 
#x and y are now ordered the same as z (increasing z) and indexsort.
xsort = np.array(xsort)
ysort = np.array(ysort)


# Flight for the magnetars, which image pixels do they land on / what are the Flight values?
mgn = 0
bestdex = []
offthehost = []
for XM in shiftedx:
       diffx = (XM - (xsort))**2
       diffy = (shiftedy[mgn] - (ysort))**2   #'shifted' includes R/2 to shift coordinates to CENTER of the pixel
       dist = np.sqrt(diffx+diffy)
       gradshifted = np.sqrt( (XM - (Res+1)/2)**2 + (shiftedy[mgn] - (Res+1)/2)**2 )
       if gradshifted >= circlecut*(Res+1):  #outside the host? then use Flight = 0
           offthehost.append(mgn)
       bestdex.append(np.where(dist == np.min(dist))[0][0])  #indicies of the cells which are closest
       mgn = mgn + 1
magnetar_flight = flight_masers[bestdex]  
magnetar_flight[offthehost] = 0

# Plotting Flight
N,bins,patches = ax1.hist(magnetar_flight,histtype='step',density=True,cumulative=True,bins=1000,linewidth=4,color='k')
patches[0].set_xy(patches[0].get_xy()[:-1])
p, = ax1.plot([0,1],[-1,-2],'-k',linewidth=4,label='Magnetars')
ax1.set_xlabel('Fraction of Light',fontsize=12)
ax1.set_ylabel('Cumulative Fraction',fontsize=12)
p, = ax1.plot([0,1],[0,1],'--k',label='Light tracer')
ax1.legend(loc=4,frameon=False)
ax1.set(xlim=(0,1), ylim=(0,1))




# Same as above, but with update-able parameters based on user defined slider values
def update(val):
    ax1.clear()
    ax2.clear()
    
    distances = MWmagnetars.Dist.values
    ymag = np.loadtxt('mcgill_y.txt') 
    xmag = np.loadtxt('mcgill_x.txt')[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]
    ymag = np.loadtxt('mcgill_y.txt')[(distances > 0) & (distances < maxdistlim) & (ymag<ylim)]   

    circlecut = s_factor.val/30  #G-centric pixel selection
    Resin = 30/(s_factor2.val) - 1
    Reslist = np.array([29,34,40,44,48,59,74,80,89,104,119])
    Diff = np.ndarray.tolist(np.abs(Resin-Reslist))
    mindex = Diff.index(np.min(Diff))
    Res = int(Reslist[mindex])

    disc_redux = s_factor3.val
    bulgebar_redux = s_factor4.val
    arm_redux = s_factor5.val
    
    arms = np.loadtxt('Fullmap/Arms_'+str(Res)+'.txt')/(arm_redux)
    disc_in = np.loadtxt('Fullmap/Disc_'+str(Res)+'.txt')/(disc_redux)
    disc = scipy.ndimage.gaussian_filter(disc_in,sigma=5)
    barbulge = np.loadtxt('Fullmap/Barbulge_'+str(Res)+'.txt')/(bulgebar_redux)
    
    Resgrid = scipy.ndimage.gaussian_filter(arms + disc + barbulge,sigma=0.75,truncate=2)
    
    R = 30/(Res+1)

    ax2.imshow(Resgrid,cmap=cm.Oranges,norm=colors.LogNorm()) 
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xticks([])
    
    rselection = s_factor.val*((Res+1)/30)
    XS = rselection*np.cos(theta) + (0/R + Res/2)
    YS = rselection*np.sin(theta) + ((8.2-8.2)/R + Res/2)
    ax2.plot(XS,YS,color='cadetblue',linestyle='--')
    ax2.set_xlim([0,Res+1])
    ax2.set_ylim([0,(Res+1)/2])
    
    shiftedx = (xmag/R + Res/2)
    shiftedy = ((ymag-8.2)/R + Res/2)
    ax2.scatter(shiftedx,shiftedy)
    ax2.plot((0/R + Res/2),((0-8.2)/R + Res/2),'ok',fillstyle='none',markersize=10)
    ax2.plot((0/R + Res/2),((0-8.2)/R + Res/2),'+k',markersize=10)
    
    ax2.invert_yaxis()
    
    zlist = []
    rlist = []
    indicies = []
    II = 0
    mask = np.ones((Res+1,Res+1))      
    indexes = np.linspace(0,Res,Res+1).astype(int)
    for valx in indexes:
        for valy in indexes[0:int((Res+1)/2)]:  
            zlist = zlist + [Resgrid[valy,valx]]  
            #plt.plot(valx,valy,'.c')    #
            rval = np.sqrt((valx-(Res+1)/2)**2 + (valy-(Res+1)/2)**2)
            rlist = rlist + [rval]
            indicies.append(II)
            if rval < circlecut*(Res+1):   
                mask[valy,valx] = np.nan
            II = II + 1
    zlist = np.array(zlist)
    rlist = np.array(rlist)
    indicies = np.array(indicies)
    
    indicies = indicies[rlist < circlecut*(Res+1)]   
    zlist = zlist[rlist < circlecut*(Res+1)]  
    zsort, indexsort = zip(*sorted(zip(zlist,indicies)))  
    
    zsort = np.array(zsort)
    flcumul = []
    cflux = 0
    for entry in zsort:
        cflux = cflux + entry
        flcumul.append(cflux)
    flight_masers = np.array(flcumul)/np.max(flcumul) 
    
    xlist = []
    ylist = []
    c = 0
    for Xx in indexes:
        for Yy in indexes[0:int((Res+1)/2)]:    
            if c in indicies:  
                xlist.append(Xx)
                ylist.append(Yy)  
            c = c + 1
    zsort, xsort = zip(*sorted(zip(zlist, xlist))) 
    zsort, ysort = zip(*sorted(zip(zlist, ylist))) 

    xsort = np.array(xsort)
    ysort = np.array(ysort)
    
    mgn = 0
    bestdex = []
    offthehost = []
    for XM in shiftedx:
           diffx = (XM - (xsort))**2
           diffy = (shiftedy[mgn] - (ysort))**2   
           dist = np.sqrt(diffx+diffy)
           gradshifted = np.sqrt( (XM - (Res+1)/2)**2 + (shiftedy[mgn] - (Res+1)/2)**2 )
           if gradshifted >= circlecut*(Res+1):  
               offthehost.append(mgn)
           bestdex.append(np.where(dist == np.min(dist))[0][0]) 
           mgn = mgn + 1
    magnetar_flight = flight_masers[bestdex]  
    magnetar_flight[offthehost] = 0
    
    N,bins,patches = ax1.hist(magnetar_flight,histtype='step',density=True,cumulative=True,bins=1000,linewidth=4,color='k')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    ax1.plot([0,1],[-1,-2],'-k',linewidth=4,label='Magnetars')
    ax1.set_xlabel('Fraction of Light',fontsize=13)
    ax1.set_ylabel('Cumulative Fraction',fontsize=13)
    ax1.plot([0,1],[0,1],'--k',label='Light tracer')
    ax1.legend(loc=4,frameon=False)
    ax1.set(xlim=(0,1), ylim=(0,1))

    print('Flight values are: ')
    print(magnetar_flight)
    
    np.savetxt('flight.txt',magnetar_flight)
    print('Saved to flight.txt.')

print('Flight values are: ')
print(magnetar_flight)

np.savetxt('flight.txt',magnetar_flight)
print('Saved to flight.txt.')

s_factor.on_changed(update)
s_factor2.on_changed(update)
s_factor3.on_changed(update)
s_factor4.on_changed(update)
s_factor5.on_changed(update)