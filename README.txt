#README

In the folder `data' you can find Galactocentric offset, F_light and enclosed flux values for the magnetar, pulsar, young pulsar, HMXB and LMXB samples,
as described in Chrimes et al. (2021). In each .txt, the columns are:
0 - Galactocentric offset [kpc]
1 - Host-normalised offset (B-band)
2 - Host-normalised offset (I-band)
3 - F_light (B-band)
4 - F_light (I-band)
5 - Enclosed flux (B-band)
6 - Enclosed flux (I-band)

#############################################################################################################################################################
Everything else in this repo is adapted from the Milky Way fraction of light code used in Chrimes et al. (2021).

!!! This is WIP and is still being updated as of May 2021 !!!

Simply clone/download this repo and run either the ipython notebook or the .py script. These both produce an interactive figure showing the 
fraction of light distribution for Milky Way magnetars, given the best estimate of their heliocentric distance. Note that unlike the version used in the paper,
there is no y<8.3kpc cut, instead the user simply varies the heliocentric distance sample selection radius.

#############################################################################################################################################################

With this notebook/script, you can vary:
(i) the heliocentric distance cut - only neutron stars inside this distance will be considered
(ii) The ratio of the bulge luminosity in the I band, to the B band (default is 5.6, this is our best guess)
(iii) The ratio of the arm luminosity in the I band, to the B band (default is 2.0, this is our best guess)
(iv) The ratio of the disc luminosity in the I band, to the B band (default is 2.7, this is our best guess)
      
Note that any (ii), (iii) and (iv) values which preserve the default ratios produce a B-band image. This is just due to how we have scaled the original luminosities in each band (e.g. in the I-band, the bulge contributes 25% of the flux), see Chrimes et al. (2021) for details. In other words, these ratios are the factor by which the I-band
flux is reduced in order to get the B-band image. To get an I-band version, simply set all three reduction factors to the same value.

You can also vary:
(v) the pixel selection radius for F_light. Varying this approximates varying the background level of an image / the surface brightness of the galaxy, as if the pixels were
being selected with e.g. source extractor.
(vi) the spatial resolution, in kpc/pixel. The default is 0.25, similar to HST resolution at z~1. The actual images are not recalculated on the fly, and 
    instead rely on pre-calculated images at each resolution. As a result, the slider moves continuously, but the resolutions won't! 

#############################################################################################################################################################

Updates coming soon:
- general tidying up of the code... 
- ability to print the F_light distribution in a more user-friendly format
- ability to choose different NS populations, e.g. XRBs and pulsars
