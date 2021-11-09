#this code composite mwisp lv/m0 maps into RGB image
#1.align lvmaps to a reference
#2.output the lv and m0 images

import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parts = 2 	#how many parts to separate the plane
dpi=1000	#quality of output png
fitspath = '/share/public/shbzhang/mosaic/'# './'
lrange = (11.75,229.75)
lspan = max(lrange)-min(lrange)
brange = (-5.25,5.25)
bspan = max(brange)-min(brange)
vrange = (-120,200)
vspan = max(vrange)-min(vrange)

do_resampleLV = 0
do_drawLV = 1
do_drawII = 1

class LinearWCS():
	def __init__(self, header, axis):
		self.wcs =  {key:header['%5s%1i' % (key,axis)] for key in ['NAXIS','CTYPE','CRVAL','CDELT','CRPIX','CROTA']}

	def pixel_to_world(self, pixel, base=0):
		return (np.array(pixel)-self.wcs['CRPIX']+1-base)*self.wcs['CDELT']+self.wcs['CRVAL']

	def world_to_pixel(self, world, base=0):
		return (np.array(world)-self.wcs['CRVAL'])/self.wcs['CDELT']+self.wcs['CRPIX']-1+base

	@property
	def axis(self):
		return self.pixel_to_world(np.arange(self.wcs['NAXIS']))

	@property
	def extent(self):
		#imshow extent
		return self.pixel_to_world([-0.5, self.wcs['NAXIS']-0.5])

if __name__ == '__main__':
	#Resample velocity axis in LVMAP of 12CO/C18O to align with 13CO
	if do_resampleLV:
		hdu12 = fits.open(fitspath+'lvmapgoodlooking/tile_U_lvmap.fits')[0]
		wcs12 = LinearWCS(hdu12.header, 2)
		hdu13 = fits.open(fitspath+'lvmapgoodlooking/tile_L_lvmap.fits')[0]
		wcs13 = LinearWCS(hdu13.header, 2)
		hdu18 = fits.open(fitspath+'lvmapgoodlooking/tile_L2_lvmap.fits')[0]
		wcs18 = LinearWCS(hdu18.header, 2)

		#resample 12/18 to 13
		v12 = wcs12.axis
		v13 = wcs13.axis
		v18 = wcs18.axis
		x = np.arange(hdu13.header['NAXIS1'])

		nan13 = np.isnan(hdu13.data)

		hdu12.data[np.isnan(hdu12.data)]=0
		interpfunc = RectBivariateSpline(v12, x, hdu12.data)
		resample = interpfunc(v13,x)
		resample[nan13] = np.nan
		hdu13.data=resample
		hdu13.writeto(fitspath+'lvmapgoodlooking/tile_U_lvmap_to13CO.fits', overwrite=True)

		hdu18.data[np.isnan(hdu18.data)]=0
		interpfunc = RectBivariateSpline(v18, x, hdu18.data)
		resample = interpfunc(v13,x)
		resample[nan13] = np.nan
		hdu13.data=resample
		hdu13.writeto(fitspath+'lvmapgoodlooking/tile_L2_lvmap_to13CO.fits', overwrite=True)



	def squareroot(img, vmin, vmax):
		#squareroot scale of image
		img = (img-vmin)/(vmax-vmin)
		img[img>1]=1
		img[img<0]=0
		img[np.isnan(img)]=1 #render nan as white
		img = img**0.5
		return img

	def figuresetting(parts, separate=True, lv=False):
		#get figuresettings
		if parts == 1: figname='Single'
		elif parts == 2: figname = 'Double'
		elif parts == 3: figname = 'Triple'
		elif parts == 4: figname = 'Quadruple'
		else: figname = 'CO'
		#ax factor must be the same for IntInt map
		if lv: axsize = (lspan/10/parts, vspan/100)
		else: axsize = (lspan/10/parts, bspan/10)
		marginleft = 0.35
		marginright = 0.25
		margintop = 0.2
		marginbottom = 0.3
		spacewidth = 0.1
		spaceheight = 0.35
		if separate:
			figsize = (axsize[0]+marginleft+marginright, axsize[1]+marginbottom+margintop)
		else:
			figsize = (axsize[0]+marginleft+marginright, axsize[1]*parts+spaceheight*(parts-1)+marginbottom+margintop)
		figadjust =dict(\
			left=marginleft/figsize[0], right=1-marginright/figsize[0],\
			bottom=marginbottom/figsize[1], top=1-margintop/figsize[1], \
			wspace=spacewidth/axsize[0], hspace=spaceheight/axsize[1])
		print('Plot %s map %s.' % ('L-V' if lv else 'IntIntensity', 'separately' if separate else 'together'))
		return figname, figsize, figadjust



	#draw a lvmap image
	if do_drawLV:
		hdu12 = fits.open(fitspath+'lvmapgoodlooking/tile_U_lvmap_to13CO.fits')[0]
		hdu13 = fits.open(fitspath+'lvmapgoodlooking/tile_L_lvmap.fits')[0]
		hdu18 = fits.open(fitspath+'lvmapgoodlooking/tile_L2_lvmap_to13CO.fits')[0]
		###Do this to hide bad channels, might be problematic###
		hdu18.data[hdu18.data>hdu13.data]=0 
		rgb = np.dstack([\
			squareroot(hdu18.data,0,0.4),\
			squareroot(hdu13.data,0,3.5),\
		 	squareroot(hdu12.data,0,9)])
		ext = [*LinearWCS(hdu12.header,1).extent, *(LinearWCS(hdu12.header,2).extent/1e3)]


		plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
		plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True

		#plot parts separately
		if parts>1:
			figname, figsize, figadjust = figuresetting(parts, separate=True, lv=True)
			for i in range(parts):
				fig,ax=plt.subplots(figsize=figsize)
				plt.subplots_adjust(**figadjust)
				ax.imshow(rgb, origin='lower', extent=ext)
				ax.set_aspect('auto')
				ax.set_xlim(min(lrange)+(i+1)*lspan/parts, min(lrange)+i*lspan/parts)
				ax.set_ylim(*vrange)
				ax.set_ylabel('Velocity (km/s)', fontdict=dict(size=5))
				ax.set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
				ax.tick_params(axis='both', labelsize=5)
				ax.tick_params(which='major', length=2)
				ax.tick_params(which='minor', length=1)
				ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
				ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
				ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
				plt.savefig('%s_lvmap_%1i.png' % (figname,i+1), dpi=dpi)
			
		#plot parts together
		figname, figsize, figadjust = figuresetting(parts, separate=False, lv=True)
		fig,ax=plt.subplots(nrows=parts, figsize=figsize)
		if parts==1: ax=[ax]
		plt.subplots_adjust(**figadjust)
		for i,a in enumerate(ax):
			a.imshow(rgb, origin='lower', extent=ext)
			a.set_aspect('auto')
			a.set_xlim(min(lrange)+(i+1)*lspan/parts, min(lrange)+i*lspan/parts)
			a.set_ylim(*vrange)
			a.set_ylabel('Velocity (km/s)', fontdict=dict(size=5))
			a.tick_params(axis='both', labelsize=5)
			a.tick_params(which='major', length=2)
			a.tick_params(which='minor', length=1)
			a.xaxis.set_major_locator(ticker.MultipleLocator(10))
			a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
			a.yaxis.set_major_locator(ticker.MultipleLocator(50))
			a.yaxis.set_minor_locator(ticker.MultipleLocator(10))
		ax[-1].set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
		plt.savefig('%s_lvmap.png' % figname, dpi=dpi)


		
	#output a integrated intensity image
	if do_drawII:
		# local/*.fits are integrated over [-30,30] km/s
		# whole/*.fits are integrated over whole velocity range (cf. L-V map)
		hdu12 = fits.open(fitspath+'whole/tile_U_m0.fits')[0]
		hdu13 = fits.open(fitspath+'whole/tile_L_m0.fits')[0]
		hdu18 = fits.open(fitspath+'whole/tile_L2_m0.fits')[0]
		###Do this to hide bad channel contaminations, might be problematic###
		hdu18.data[hdu18.data>hdu13.data]=0
		rgb = np.dstack([\
			squareroot(hdu18.data,0.3,2.5),\
			squareroot(hdu13.data,0,18),\
			squareroot(hdu12.data,0,40)])
		ext = [*LinearWCS(hdu12.header,1).extent, *(LinearWCS(hdu12.header,2).extent)]
		
		plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
		plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True

		#plot parts separately
		if parts>1:
			figname, figsize, figadjust = figuresetting(parts, separate=True)
			for i in range(parts):
				fig,ax=plt.subplots(figsize=figsize)
				plt.subplots_adjust(**figadjust)
				ax.imshow(rgb, origin='lower', extent=ext)
				ax.set_aspect('equal')
				ax.set_xlim(min(lrange)+(i+1)*lspan/parts, min(lrange)+i*lspan/parts)
				ax.set_ylim(*brange)
				ax.set_ylabel('Galactic Latitude (deg)', fontdict=dict(size=5))
				ax.set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
				ax.tick_params(axis='both', labelsize=5)
				ax.tick_params(which='major', length=2)
				ax.tick_params(which='minor', length=1)
				ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
				ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
				ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
				plt.savefig('%s_%1i.png' % (figname,i+1), dpi=dpi)
			
		#plot parts together
		figname, figsize, figadjust = figuresetting(parts, separate=False)
		fig,ax=plt.subplots(nrows=parts, figsize=figsize)
		if parts==1: ax=[ax]
		plt.subplots_adjust(**figadjust)
		for i,a in enumerate(ax):
			a.imshow(rgb, origin='lower', extent=ext)
			a.set_aspect('equal')
			a.set_xlim(min(lrange)+(i+1)*lspan/parts, min(lrange)+i*lspan/parts)
			a.set_ylim(*brange)
			a.set_ylabel('Galactic Latitude (deg)', fontdict=dict(size=5))
			a.tick_params(axis='both', labelsize=5)
			a.tick_params(which='major', length=2)
			a.tick_params(which='minor', length=1)
			a.xaxis.set_major_locator(ticker.MultipleLocator(10))
			a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
			a.yaxis.set_major_locator(ticker.MultipleLocator(2))
			a.yaxis.set_minor_locator(ticker.MultipleLocator(1))
		ax[-1].set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
		plt.savefig('%s.png' % figname, dpi=dpi)
