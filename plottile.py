#align lvmaps to a reference
#output the lv and ii images

import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


if 1:
	#Resample LVMAP of 12CO/C18O to 13CO
	hdu12 = fits.open('lvmapgoodlooking/tile_U_lvmap.fits')[0]
	wcs12 = LinearWCS(hdu12.header, 2)
	hdu13 = fits.open('lvmapgoodlooking/tile_L_lvmap.fits')[0]
	wcs13 = LinearWCS(hdu13.header, 2)
	hdu18 = fits.open('lvmapgoodlooking/tile_L2_lvmap.fits')[0]
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
	hdu13.writeto('lvmapgoodlooking/tile_U_lvmap_to13CO.fits', overwrite=True)

	hdu18.data[np.isnan(hdu18.data)]=0
	interpfunc = RectBivariateSpline(v18, x, hdu18.data)
	resample = interpfunc(v13,x)
	resample[nan13] = np.nan
	hdu13.data=resample
	hdu13.writeto('lvmapgoodlooking/tile_L2_lvmap_to13CO.fits', overwrite=True)

def squareroot(img, vmin, vmax):
	#squareroot scale of image
	img = (img-vmin)/(vmax-vmin)
	img[img>1]=1
	img[img<0]=0
	img[np.isnan(img)]=1 #render nan as white
	img = img**0.5
	return img

if 1:
	#output a lvmap image
	hdu12 = fits.open('lvmapgoodlooking/tile_U_lvmap_to13CO.fits')[0]
	hdu13 = fits.open('lvmapgoodlooking/tile_L_lvmap.fits')[0]
	hdu18 = fits.open('lvmapgoodlooking/tile_L2_lvmap_to13CO.fits')[0]
	hdu18.data[hdu18.data>hdu13.data]=0 #hide bad channels
	rgb = np.dstack([\
		squareroot(hdu18.data,0,0.4),\
		squareroot(hdu13.data,0,3.5),\
	 	squareroot(hdu12.data,0,9)])
	ext = [*LinearWCS(hdu12.header,1).extent, *(LinearWCS(hdu12.header,2).extent/1e3)]

	plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
	plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True
	fig,ax=plt.subplots(figsize=(30,5))
	#fig.tight_layout()
	plt.subplots_adjust(left=0.015, right=0.99, bottom=0.08, top=0.95)
	ax.imshow(rgb, origin='lower', extent=ext)
	ax.set_aspect('auto')
	ax.set_xlim(229.75,11.75)
	ax.set_ylim([-120,200])
	ax.set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
	ax.set_ylabel('Velocity (km/s)', fontdict=dict(size=5))
	ax.tick_params(axis='both', labelsize=5)
	ax.tick_params(which='major', length=2)
	ax.tick_params(which='minor', length=1)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
	plt.savefig('LV.png',dpi=300)
	#plt.show()
	
if 1:
	#output a integrated intensity image
	hdu12 = fits.open('whole/tile_U_m0.fits')[0]
	hdu13 = fits.open('whole/tile_L_m0.fits')[0]
	hdu18 = fits.open('whole/tile_L2_m0.fits')[0]
	hdu18.data[hdu18.data>hdu13.data]=0 #clean
	rgb = np.dstack([\
		squareroot(hdu18.data,0.3,2.5),\
		squareroot(hdu13.data,0,18),\
		squareroot(hdu12.data,0,40)])
	ext = [*LinearWCS(hdu12.header,1).extent, *(LinearWCS(hdu12.header,2).extent)]
	
	plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
	plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True
	fig,ax=plt.subplots(figsize=(30,2))
	#fig.tight_layout()
	plt.subplots_adjust(left=0.015, right=0.99, bottom=0.01, top=0.99)
	ax.imshow(rgb, origin='lower', extent=ext)
	ax.set_aspect('equal')
	ax.set_xlim(229.75,11.75)
	ax.set_ylim([-5.25,5.25])
	ax.set_xlabel('Galactic Longitude (deg)', fontdict=dict(size=5))
	ax.set_ylabel('Galactic Latitude (deg)', fontdict=dict(size=5))
	ax.tick_params(axis='both', labelsize=5)
	ax.tick_params(which='major', length=2)
	ax.tick_params(which='minor', length=1)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
	plt.savefig('CO.png',dpi=400)
