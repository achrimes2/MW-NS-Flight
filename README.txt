#README

Some code adapted from the neutron star - Milky Way fraction of light code used for Chrimes et al. (2021).

!!! This is WIP and is still being updated as of April 2021 !!!

Simply clone/download this repo and run either the ipython notebook or the .py script. These both produce an interactive figure showing the 
fraction of light distribution for Milky Way magnetars, given the best estimate of their heliocentric distance. 

#############################################################################################################################################################

With this notebook/script, you can vary:
(i) the heliocentric distance cut - only neutron stars inside this distance will be considered
(ii) The ratio of the bulge luminosity in the I band, to the B band (default is 6, this is our best guess)
(iii) The ratio of the disc luminosity in the I band, to the B band (default is 3, this is our best guess)
      
Note that using (i) and (ii) values in any 1:2 ratio produces a B-band image. Making the ratio 1:1 gives you an I-band image.
This is just due to how we have scaled the original luminosities in each band (e.g. in the I-band, the bulge contributes 25% of the flux), 
see Chrimes et al. (2021) for details.

You can also vary:
(iv) the pixel selection radius for F_light. Varying this simulates (albeit crudely) varying the background level of an image / the surface brightness of the galaxy.
(v) the spatial resolution, in kpc/pixel. The default is 0.25, similar to HST resolution at z~1. The actual images are not recalculated on the fly, and 
    instead rely on pre-calculated images at each resolution. Currently only a few resolutions are currently available, so the slider may move continuously, but the 
    resolution won't! 

#############################################################################################################################################################

Updates coming soon:
- general tidying up of the code...
- more spatial resolutions
- ability to print the F_light distribution
- ability to choose different NS populations, e.g. XRBs and pulsars
