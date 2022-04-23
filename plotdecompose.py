import os,glob,sys,getpass
import astropy.units as u
import astropy.constants as con
from astropy.units.quantity import Quantity
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
sys.path.append(os.path.abspath('../DeepOutflow/procedure/'))
from regulatetable import Catalogue, Sample
_user = getpass.getuser()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib import cm
dtor=np.pi/180
font = dict(SMALL = 8, MEDIUM = 10, LARGE = 12)


def myrainbow(turn=100, final=0.85):
	old = cm.get_cmap('rainbow', 256)
	new = old(np.linspace(0,1,256))
	#new[:,1]=new[:,1]**0.9
	#new[:,2]=new[:,2]*0.8
	new = final-new
	new[:turn,:3] *= (np.linspace(0,1,turn)**2)[:,np.newaxis]
	new = final-new
	new[new>1]=1
	return ListedColormap(new)

def myBlRd(Vw = 50):
	my = cm.get_cmap('RdBu_r', 256)
	new = my(np.linspace(0,1,256))
	shapeV = np.hstack([np.ones(128-Vw), np.linspace(1,0,Vw), np.linspace(0,1,Vw), np.ones(128-Vw)])
	shapeV = shapeV**0.6
	new[:,:3] *= (shapeV*0.3+0.7)[:,np.newaxis]
	new[:,:3] /= new[:,:3].max()
	return ListedColormap(new)

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

class Parameter(Quantity):
	def __new__(cls, *args, bins=None, label=None, resolution=None, error=None, meanbins=None, scale='log', **kwargs):
		obj = super().__new__(cls, *args, **kwargs)
		obj.bins = bins
		obj.label = label
		obj.resolution = resolution
		obj.error = error
		obj.meanbins = meanbins
		obj.scale = scale
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		super().__array_finalize__(obj)
		self.__dict__['bins'] = getattr(obj, 'bins', None)
		self.__dict__['label'] = getattr(obj, 'label', None)
		self.__dict__['resolution'] = getattr(obj, 'resolution', None)
		self.__dict__['error'] = getattr(obj, 'error', None)
		self.__dict__['meanbins'] = getattr(obj, 'meanbins', None)
		self.__dict__['scale'] = getattr(obj, 'scale', None)

	@property
	def axislabel(self):
		label = '' if self.label is None else self.label
		unit = '' if (self.unit is None) | (self.unit==u.Unit()) else '[%s]' % self.unit.to_string('latex_inline')
		if label=='': return unit
		if unit=='': return label
		return label+' '+unit

	@property
	def lim(self):
		return Parameter([np.nanmin(self.value), np.nanmedian(self.value), np.nanmax(self.value)], \
			unit=self.unit, label=self.label)

#Read all properties
D = Parameter(np.load('D.npy'), unit=u.kpc, label='Distance')
l = Parameter(np.load('l.npy'), unit=u.deg, label='Galactic Longitude')
b = Parameter(np.load('b.npy'), unit=u.deg, label='Galactic Latitude')

#the size of half maximum
R = Parameter(np.load('R.npy'), unit=u.pc, label=r'R')
angsz = Parameter(np.load('angsz.npy'), unit=u.rad, label = 'Angular Size')

area = Parameter(np.load('area.npy'), label=r'Area (pixel)')
#the size of lowest contour
Rboundary = Parameter(np.sqrt(np.load('area.npy')/np.pi)*30, unit=u.arcsec, resolution=52/3600*dtor * D *1e3, \
	bins=10**np.linspace(-2.4, 1.7, 200), label=r'R')
#Rboundary = Parameter(np.sqrt(np.load('area.npy')/np.pi)*30/3600/180*np.pi*D.to('pc').value, unit=u.pc, resolution=52/3600*dtor * D *1e3, \
#	bins=10**np.linspace(-2.4, 1.7, 200), label=r'R [pc]')

SD = Parameter(np.load('SD.npy'), unit=u.Msun/u.pc**2, label=r'$\Sigma$')
mass = Parameter(np.load('mass.npy'), unit=u.Msun, label=r'M$_{LTE}$')
Tex = Parameter(np.load('Tex.npy'), unit=u.K, label=r'T$_{ex}$')
vdensity = Parameter(np.load('n.npy'), unit=u.cm**-3, label=r'n')

v0 = Parameter(np.load('avm1.npy'), unit=u.km/u.s, label=r'$v_{LSR}$')
vd = Parameter(np.load('avm2.npy'), unit=u.km/u.s, label=r'$\sigma_v$')
tvd = Parameter(np.load('tvd.npy'), unit=u.km/u.s, label=r'$\sigma_{th}$')
ntvd = Parameter(np.load('ntvd.npy')[:,1], unit=u.km/u.s, label=r'Averaged $\sigma_{nt}$ of spectra')
avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, label=r'$\sigma_{nt}$ of averaged spectrum')

class PlotCatalogue():
	def __init__(self):
		self.catalogue = 'clump_mas.cat'
		self.lrange = (11.75,229.75)
		self.brange = (-5.25,5.25)
		self.dpi = 500
		self.lbmap_parts = 3

		plt.rcParams['xtick.top']=True
		plt.rcParams['ytick.right']=True
		plt.rcParams['xtick.direction']=plt.rcParams['ytick.direction']='in'
		plt.rcParams['xtick.labelsize']=plt.rcParams['ytick.labelsize']=font['SMALL']
		plt.rcParams['xtick.major.size']=plt.rcParams['ytick.major.size']=4
		plt.rcParams['xtick.minor.size']=plt.rcParams['ytick.minor.size']=2

	def _squareroot(img, vmin, vmax):
		#squareroot scale of image
		img = (img-vmin)/(vmax-vmin)
		img[img>1]=1
		img[img<0]=0
		img[np.isnan(img)]=1 #render nan as white
		img = img**0.5
		return img

	def _figuresetting(xspan=None, yspan=None, parts=1, separate=False, lv=False, colorbar=True):
		#get figuresettings
		#ax factor must be the same for IntInt map
		if lv: axsize = (xspan/10/parts, yspan/100)
		else: axsize = (xspan/10/parts, yspan/10)
		marginleft = 0.55
		marginright = 1.15 if colorbar else 0.5
		margintop = 0.3
		marginbottom = 0.45
		spacewidth = 0.1
		spaceheight = 0.37
		#colorbarwidth = 0 if colorbar else 0
		if separate:
			figsize = (axsize[0]+marginleft+marginright, axsize[1]+marginbottom+margintop)
		else:
			figsize = (axsize[0]+marginleft+marginright, axsize[1]*parts+spaceheight*(parts-1)+marginbottom+margintop)
		figadjust =dict(\
			left=marginleft/figsize[0], right=1-marginright/figsize[0],\
			bottom=marginbottom/figsize[1], top=1-margintop/figsize[1], \
			wspace=spacewidth/axsize[0], hspace=spaceheight/axsize[1])
		return figsize, figadjust

	def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list(
			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
			cmap(np.linspace(minval, maxval, n)))
		return new_cmap

	##################################################################################################################
	def _lbmap(lrange, brange, figname=None, parts=1, dpi=600, layers=[]):
		xspan = max(lrange)-min(lrange)
		yspan = max(brange)-min(brange)

		plt.rcParams['xtick.labeltop']=plt.rcParams['ytick.labelright']=True
		plt.rcParams['xtick.major.size']=plt.rcParams['ytick.major.size']=2
		plt.rcParams['xtick.minor.size']=plt.rcParams['ytick.minor.size']=1
		plt.rcParams['image.aspect']='equal'

		colorbar = any(['colorbar' in l for l in layers])
		figsize, figadjust = PlotCatalogue._figuresetting(xspan=xspan, yspan=yspan, parts=parts, separate=False, colorbar=colorbar)
		fig,ax=plt.subplots(nrows=parts, figsize=figsize)
		if parts==1: ax=[ax]
		plt.subplots_adjust(**figadjust)

		for i,a in enumerate(ax):
			for l in layers:
				method = eval('a.'+l['method'])
				if 'colorbar' in l:
					im = method(*l['args'], **l['kws'])
					cbkws = l['colorbar']
				else:
					method(*l['args'], **l['kws'])

			a.set_xlim(min(lrange)+(i+1)*xspan/parts, min(lrange)+i*xspan/parts)
			a.set_ylim(*brange)
			a.xaxis.set_major_locator(ticker.MultipleLocator(10))
			a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
			a.yaxis.set_major_locator(ticker.MultipleLocator(2))
			a.yaxis.set_minor_locator(ticker.MultipleLocator(1))
			a.set_aspect('equal')
			a.tick_params(axis='both', labelsize=font['SMALL'])
		ax[1].set_ylabel('Galactic Latitude (deg)', fontsize=font['MEDIUM'])
		ax[-1].set_xlabel('Galactic Longitude (deg)', fontsize=font['MEDIUM'])

		if colorbar:
			#cb = fig.colorbar(im, ax=ax.ravel().tolist(), aspect=35)
			#cb.set_label(**cbkws)
			#cb.ax.tick_params(axis='both', labelsize=5)
			cb_ax = fig.add_axes([1-0.75/figsize[0], figadjust['bottom'], 0.01, figadjust['top']-figadjust['bottom']])
			cbar = fig.colorbar(im, cax=cb_ax)
			#cb_ax.text(1, 1., cbkws['label'], va='bottom', ha='right', transform=cb_ax.transAxes, fontsize=font['SMALL'])
			cbar.set_label(**cbkws, fontsize=font['MEDIUM'])
			cbar.ax.tick_params(axis='both', labelsize=font['SMALL'])

			#axin = inset_axes(ax[0], width='2%', height='300%', loc='lower right', \
			#	bbox_to_anchor=(0.10,0.08,1,1), bbox_transform=ax[-1].transAxes)
			#cb = fig.colorbar(im, cax=axin, orientation='vertical', pad=0.05)
			#axin.text(-1, 1.03, cbkws['label'], transform=axin.transAxes, fontsize=font['SMALL'])
			#cb.ax.tick_params(axis='both', labelsize=font['SMALL'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_lbmap(self, n=[0,]):
		layers=[]
		l = Parameter(np.load('l.npy'), unit=u.deg, label='Galactic Longitude')
		b = Parameter(np.load('b.npy'), unit=u.deg, label='Galactic Latitude')
		if 0 in n:
			figname = 'fig_lbmap_vs'
			hdu = fits.open('/Users/sz268601/Work/GuideMap/whole/tile_L_m0.fits')[0]
			img = PlotCatalogue._squareroot(hdu.data,0,18)
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,), \
				'kws':dict(origin='lower', extent=ext, cmap='gray')})


			v0 = Parameter(np.load('avm1.npy'), unit=u.km/u.s, label=r'$v_{LSR}$')
			s = Parameter(np.load('angsz.npy'), unit=u.rad, label=r'Angular Size')
			#avntvd = np.load('avntvd.npy')
			#R = np.load('R.npy')
			#D = np.load('D.npy')
			#Tex = np.load('Tex.npy')
			#idx = (R>1) & (avntvd<0.5) & (D<3) & (Tex<20) & (l<105*u.deg)
			cmap = PlotCatalogue._truncate_colormap(plt.get_cmap('gist_rainbow'), 0.8, 0.)
			layers.append({'method':'scatter', 'args':(l, b), \
				'kws':dict(c=v0, vmin=-50, vmax=50, cmap=cmap, s=s.to(u.deg)**2*200, \
				marker='.', edgecolors='none', alpha=0.4),\
				'colorbar':dict(label=v0.axislabel)})
			#layers.append({'method':'plot', 'args':(l[idx], b[idx], 'r.'), \
			#	'kws':dict(markersize=1, alpha=1)})
			#3300/16=1deg
			#3300=4deg
			#layers.append({'method':'scatter', 'args':([229.75,], [0,]), \
			#	'kws':dict(c=[0,], s=[3300,], \
			#	marker='.', edgecolors='none', alpha=0.7),\
			#	'colorbar':dict(label=v0.axislabel)})

		if 1 in n:
			figname = 'fig_lbmap_cnt1'
			#hdu = fits.open('/Users/sz268601/Work/GuideMap/local/tile_L_m0.fits')[0]
			hdu = fits.open('N2kpc.fits')[0]
			hdu.data[np.isnan(hdu.data)] = 0
			#import scipy.ndimage as ndimage
			#img = ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0)
			img = PlotCatalogue._squareroot(hdu.data,0,0.1)
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,), \
				'kws':dict(origin='lower', extent=ext, cmap='gray', vmin=-5, vmax=1)})

			D = Parameter(np.load('D.npy'), unit=u.kpc, label='Distance')
			#v0 = Parameter(np.load('avm1.npy'), unit=u.km/u.s, label=r'$v_{LSR}$')
			idx = D<2*u.kpc#np.abs(v0)<30*u.km/u.s#
			avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, label=r'$\sigma_{nt}$')
			layers.append({'method':'scatter', 'args':(l[idx], b[idx]), \
				'kws':dict(c=np.log10(avntvd[idx].value), vmin=-0.63, vmax=0, cmap='jet', s=1.5, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='log(%s)' % avntvd.axislabel)})
 
		if 2 in n:
			figname = 'fig_lbmap_ntmap_whole'
			hdu = fits.open('Bfield/nt2kpc_fwhm7.fits')[0]
			nan = np.isnan(hdu.data)
			hdu.data[nan] = 0.17/2.355
			img = np.log10(hdu.data)
			#import scipy.ndimage as ndimage
			#img = np.log10(ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0))
			#img[nan]=-1#np.nan
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, label=r'$\sigma_{nt}$')
			layers.append({'method':'imshow', 'args':(img,),\
				'kws':dict(origin='lower', extent=ext, vmin=-0.63, vmax=0, cmap='jet'), \
				'colorbar':dict(label='log(%s)' % avntvd.axislabel)})

		if 3 in n:
			figname = 'fig_lbmap_Smap'
			hdu = fits.open('Bfield/interp_nearest_mapS_fwhm7_ns2048_AngSt1.fits')[0]
			#import scipy.ndimage as ndimage
			#nan = np.isnan(hdu.data)
			#hdu.data[nan]=0.17/2.355
			img = np.log10(hdu.data)
			#import scipy.ndimage as ndimage
			#img = np.log10(ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0))
			#img[nan]=-1#np.nan
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,),\
				'kws':dict(origin='lower', extent=ext, vmin=0, vmax=1.9, cmap='jet'), \
				'colorbar':dict(label=r'log(S [$^\circ$])')})

		if 4 in n:
			figname = 'fig_lbmap_Spmap'
			hdu = fits.open('Bfield/interp_nearest_mapSp_fwhm7_ns2048_AngSt1.fits')[0]
			#import scipy.ndimage as ndimage
			#nan = np.isnan(hdu.data)
			#hdu.data[nan]=0.17/2.355
			img = np.log10(hdu.data)
			#import scipy.ndimage as ndimage
			#img = np.log10(ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0))
			#img[nan]=-1#np.nan
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,),\
				'kws':dict(origin='lower', extent=ext, vmin=-1.2, vmax=-0.2, cmap='jet'), \
				'colorbar':dict(label=r'log(S p [$^\circ$])')})

		PlotCatalogue._lbmap(self.lrange, self.brange, figname=figname, parts=self.lbmap_parts, dpi=self.dpi, layers=layers)


	##################################################################################################################
	def _gplane(xrange=[-7,15], yrange=[-11,17], figname=None, dpi=600, layers=[], R0=8.15):
		plt.rcParams['xtick.labeltop']=plt.rcParams['ytick.labelright']=True
		plt.rcParams['image.aspect']='equal'

		colorbar = any(['colorbar' in l for l in layers])
		fig,ax=plt.subplots(figsize=[6,7.5])
		plt.subplots_adjust(left=0.12,right=0.92,top=0.96,bottom=0.08)
		
		#
		for adeg in np.arange(0,360,15):
			arad = adeg*np.pi/180
			ax.plot([0.5*np.sin(arad), 35*np.sin(arad)], [R0-0.5*np.cos(arad), R0-35*np.cos(arad)], \
				color='silver',linewidth=0.4, linestyle=(0, (5, 5)))
			#draw gl at boundary
			dst = [x/np.sin(arad) for x in xrange] + [(R0-y)/np.cos(arad) for y in yrange]
			dst = [d if d>0 else np.inf for d in dst]
			idx = np.argmin(dst)
			if idx==0:
				tx = xrange[0]+0.3
				ty = R0-tx/np.tan(arad)
				ha,va = 'left','center'
			if idx==1:
				tx = xrange[1]-0.3
				ty = R0-tx/np.tan(arad)
				ha,va = 'right','center'
			if idx==2:
				ty = yrange[0]+0.3
				tx = (R0-ty)*np.tan(arad)
				if adeg==0: ha,va='center','bottom'
				elif adeg<180: ha,va='right','center'
				else: ha,va='left','center'
			if idx==3:
				ty = yrange[1]-0.3
				tx = (R0-ty)*np.tan(arad)
				if adeg==180: ha,va='center','top'
				elif adeg<180: ha,va='right','center'
				else: ha,va='left','center'
			#ha = 'right' if adeg<=180 else 'left'
			if adeg%180==0: rotation=0
			elif adeg<180: rotation = adeg-90
			else: rotation = adeg+90
			ax.text(tx, ty, ('l=' if adeg==0 else '') + '%i$^o$' % adeg, \
				rotation=rotation, rotation_mode='anchor', \
				horizontalalignment=ha, verticalalignment=va, color='silver', fontsize=font['SMALL'])

		#rays from sun
		#ax.plot([0, 20*np.sin(12.0*dtor)], [R0, R0-20*np.cos(12.0*dtor)], '--', color='red')
		#ax.plot([0, 20*np.sin(230*dtor)], [R0, R0-20*np.cos(230*dtor)], '--', color='red')
		#ax.plot([0, 10*np.sin(306.1*dtor)], [R0, R0-10*np.cos(306.1*dtor)], '--', color='blue')
		#ax.plot([0, 10*np.sin(285.6*dtor)], [R0, R0-10*np.cos(285.6*dtor)], '--', color='magenta')
		
		###grid
		#for x in range(-10,11,5):
		#	plt.plot([x,x],[-20,20],'--',color='grey')
		#	plt.plot([-20,20],[x,x],'--',color='grey')
		#plt.plot([-20,20],[12,12],'--',color='grey')

		for l in layers:
			method = eval('ax.'+l['method'])
			if 'colorbar' in l:
				im = method(*l['args'], **l['kws'])
				cbkws = l['colorbar']
			else:
				method(*l['args'], **l['kws'])

		def circ(xc,yc,r):
			theta = np.linspace(0,np.pi*2,361)
			return r*np.cos(theta)+xc, r*np.sin(theta)+yc
		ax.plot(*circ(0,0,0.1), 'k',linewidth=0.6)
		ax.text(-0.1,0.1,'GC', fontsize=font['SMALL'], horizontalalignment='right',color='dimgray')
		#ax.plot(*circ(0,0,5),'--', linewidth=0.6)
		#ax.plot(*circ(0,0,R0),'--', linewidth=0.6)
		#ax.plot(*circ(0,R0/2,R0/2),'--', linewidth=0.6)

		def spiral(ax, brange, bk, Rk, phi_lt, phi_gt, width=0, label='', **kws):
			#brange=[180-230,180-12]
			kws.update(dict(linewidth=0.6, alpha=1, color='dimgray'))
			b=np.arange(*brange,0.3)*dtor
			bk *= dtor
			phi = np.array([phi_lt if v<=bk else phi_gt for v in b])*dtor
			for rk in [Rk,]:# Rk-width*1.1775, Rk+width*1.1775]:
				R = rk*np.exp(-(b-bk)*np.tan(phi))
				x=R*np.sin(b)
				y=R*np.cos(b)
				ax.plot(x, y, '-' if rk==Rk else '--', **kws)
				#ax.plot(Rk*np.sin(bk), Rk*np.cos(bk),'+', **kws)
			from curvetext import CurvedText
			CurvedText(x,y,label,va='bottom', axes=ax, fontsize=font['SMALL'], color=kws['color'])
		###Parameters in Reid2019
		#spiral(ax, [-150,190],15, 3.52, -4.2, -4.2, width=0.18, color='yellow')	#3kpc
		#spiral(ax, [-60,54],  18, 4.46, -1.0, 19.5, width=0.14, color='red')	#Norma
		#spiral(ax, [-330,81], 23, 4.91, 14.1, 12.1, width=0.23, color='blue')	#Sct-Cen
		#spiral(ax, [-150,230],24, 6.04, 17.1,  1.0, width=0.27, color='magenta')#Sgt-Car
		#spiral(ax, [-30,34],   9, 8.26, 11.4, 11.4, width=0.31, color='cyan')	#local
		#spiral(ax, [-30,255], 40, 8.87, 10.3,  8.7, width=0.35, color='black')	#Perseus
		#spiral(ax, [-30,270], 18,12.24,  3.0,  9.4, width=0.65, color='red')	#Outer
		###Parameters adjusted
		#spiral(ax, [-140,18],15, 3.52, -4.2, -4.2, width=0.18, color='yellow')	#3kpc
		#spiral(ax, [45,200],90, 3.3, 4.2, 4.2, width=0.18, color='yellow')	#3kpc
		#spiral(ax, [-55,55],  18, 4.46, -0.5, 19.5, width=0.14, color='red')	#Norma
		#spiral(ax, [-54,81], 23, 4.91, 14.1, 12.1, width=0.23, color='blue')	#Sct-Cen
		#spiral(ax, [-340,-53],-54, 6.88, 10.0, 12.1, width=0.23, color='blue')
		#spiral(ax, [-33,230], 24, 6.04, 16.2,  7.1, width=0.27, color='magenta')#Sgt-Car
		#spiral(ax, [-170,-32],-33, 8.20, 9.9,  7.1, width=0.27, color='magenta')
		#spiral(ax, [-30,34],   9, 8.26, 11.4, 11.4, width=0.31, color='cyan')	#local
		#spiral(ax, [-30,255], 40, 8.87, 10.3, 13.0, width=0.35, color='black')	#Perseus
		#spiral(ax, [-30,305], 18,12.24,  4.9, 11.6, width=0.65, color='red')	#Outer
		###Paramters for me
		#spiral(ax, [-140,18],15, 3.52, -4.2, -4.2, width=0.18, label='3kpc', color='yellow')	#3kpc
		#spiral(ax, [45,200], 90,  3.3,  4.2,  4.2, width=0.18, color='yellow')	#3kpc
		spiral(ax, [-20,55], 18, 4.46, -0.5, 19.5, width=0.14, label='Norma', color='darkred')	#Norma
		spiral(ax, [-47,81], 23, 4.91, 14.1, 12.1, width=0.23, label='Scutum–Centaurus', color='darkblue')
		spiral(ax, [-342,-192],-54, 6.88, 10.0, 12.1, width=0.23, label='Outer–Scutum–Centaurus', color='darkblue')
		spiral(ax, [-24,168],24, 6.04, 16.2,  7.1, width=0.27, label='Sagittarius', color='Purple')#Sgt-Car
		spiral(ax, [-19,34],  9, 8.26, 11.4, 11.4, width=0.31, label='Local', color='darkcyan')	#local
		spiral(ax, [-26,168],40, 8.87, 10.3, 13.0, width=0.35, label='Perseus', color='black')	#Perseus
		spiral(ax, [-27,168],18,12.24,  4.9, 11.6, width=0.65, label='Outer', color='darkred')	#Outer

		#spiral(ax, [0, 150],43.9, 16.1, 9.3, 9.3, linewidth=0.5, color='green')	#Sun Yan

		ax.set_xlim(*xrange)
		ax.set_ylim(*yrange)
		ax.set_aspect('equal')
		ax.tick_params(axis='both', labelsize=font['SMALL'])
		ax.set_xlabel('X (kpc)', fontsize=font['MEDIUM'])
		ax.set_ylabel('Y (kpc)', fontsize=font['MEDIUM'])
		ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

		if colorbar:
			axin = inset_axes(ax, width='2.7%', height='35%', loc='lower left', \
				bbox_to_anchor=(0.12,0.06,1,1), bbox_transform=ax.transAxes)
			cb = fig.colorbar(im, cax=axin, orientation='vertical', pad=0.05)
			axin.text(1, 1.05, cbkws['label'], transform=axin.transAxes, fontsize=font['SMALL'], horizontalalignment='center')
			cb.ax.tick_params(axis='both', labelsize=font['SMALL'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_gplane(self, n=(3,)):
		#PlotCatalogue._gplane(dpi=self.dpi)
		#return
		#cat = Catalogue().open('clump_self0.10_equalusenear.cat')
		#cat = cat[np.argsort(cat.SurfaceDensity)]
		suffix = ''#'_fromSelf0.10eun'

		D = Parameter(np.load('D.npy'), unit=u.kpc, label='Distance')
		l = Parameter(np.load('l.npy'), unit=u.deg, label='Galactic Longitude')
		b = Parameter(np.load('b.npy'), unit=u.deg, label='Galactic Latitude')
		R0=8.15

		x = D*np.cos(b)*np.cos(l-90*u.deg)
		y = D*np.cos(b)*np.sin(l-90*u.deg)+R0*u.kpc
		x.label='X'
		y.label='Y'
		R = Parameter(np.load('R.npy'), unit=u.pc, label=r'R')
		s = (np.log10(R.value)+3)**2*0.3
		#s = (R.value)**2*0.3

		if 0 in n:
			#simple distribution
			layers=[]
			figname = 'fig_plane_xys'+suffix
			#layers.append({'method':'plot', 'args':(x, y, '.'), \
			#	'kws':dict(markersize=0.2)})
			value = R
			print(np.log10(value.lim.value))
			idx = np.argsort(value)
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=-1, vmax=1, s=5, \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 1 in n:
			#color=mass, size=size
			value = Parameter(np.load('mass.npy'), unit=u.Msun, label=r'Mass')
			idx = np.argsort(value)
			layers=[]
			figname = 'fig_plane_ms'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=1, vmax=4.5, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			'''
			###log scale colorbar
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=value[idx], cmap='rainbow', norm=LogNorm(1e1, 3e4), s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label=value.axislabel)})
			'''
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 2 in n:
			#color=Surfacedensity, size=size
			value = Parameter(np.load('SD.npy'), unit=u.Msun/u.pc**2, label=r'$\Sigma$')
			idx = np.argsort(value)
			layers=[]
			figname = 'fig_plane_SDs'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=0.3, vmax=1.8, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 3 in n:
			#color=VelocityDispersion
			value = Parameter(np.load('avm2.npy'), unit=u.km/u.s, label=r'$\sigma_v$')
			idx = np.argsort(value)
			layers=[]
			figname = 'fig_plane_vds'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=-0.8, vmax=0.4, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 4 in n:
			#color=Temperature
			value = Parameter(np.load('Tex.npy'), unit=u.K, label=r'T$_{ex}$')
			idx = np.argsort(value)
			layers=[]
			figname = 'fig_plane_Ts'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=value[idx].value, cmap=myrainbow(), vmin=5, vmax=25, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label=value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)


		if 5 in n:
			#color=Temperature
			value = Parameter(np.load('n.npy'), unit=u.cm**-3, label=r'n')
			idx = np.argsort(value)
			layers=[]
			figname = 'fig_plane_ns'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=1, vmax=4, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 6 in n:
			#color=vertical distance
			value = (D*np.sin(b))#.to(u.pc)
			value.label = 'Z'
			layers=[]
			figname = 'fig_plane_Zs'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=value.value, cmap=myBlRd(), vmin=-0.2, vmax=0.2, s=s, \
				marker='.', edgecolors='none', alpha=0.7), \
				'colorbar':dict(label=value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 7 in n:
			#color=VelocityDispersion
			tvd = Parameter(np.load('tvd.npy'), unit=u.km/u.s, label=r'$\sigma_{th}$')
			avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, label=r'$\sigma_{nt}$')
			value = Parameter(avntvd.value/tvd.value, label=r'Mach Number')
			idx = np.argsort(value)
			layers=[]
			figname = None#'fig_plane_vds'+suffix
			layers.append({'method':'scatter', 'args':(x[idx], y[idx]), \
				'kws':dict(c=np.log10(value[idx].value), cmap=myrainbow(), vmin=-0.2, vmax=1, s=s[idx], \
				marker='.', edgecolors='none', alpha=0.8), \
				'colorbar':dict(label='log(%s)' % value.axislabel)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)


	##################################################################################################################
	def plot_point3(self, figname, dpi=400):
		fig,ax=plt.subplots(ncols=3,sharex=True,sharey=True,figsize=[12,4])
		plt.subplots_adjust(left=0.1,right=0.95,top=0.92,bottom=0.12)

		def tmp(ax, cat, Dist):
			X = cat.angsz/2*Dist*1e3 * cat.SurfaceDensity
			Y = cat.avm2#np.sqrt(3)

			inner = (cat.Rgal<8.15)
			usenear = cat.D==cat.Dnear
			print(usenear.sum())
			#usenear = (np.log10(cat.SurfaceDensity)<0.7) & (cat.avm2<1) & (cat.angsz>0.0008) & np.isfinite(cat.Dnear) & (cat.D==cat.Dfar) & (cat.D>8)
			#usenear = ~usenear
			ax.plot(X[usenear & inner], Y[usenear & inner],'.',markersize=0.5,color='orangered')
			ax.plot(X[~usenear & inner], Y[~usenear & inner],'.',markersize=0.5,color='royalblue',alpha=0.8)

			X = cat.angsz/2*cat.D*1e3 * cat.SurfaceDensity
			#outer = ~inner & np.isfinite(X)
			reference = np.isfinite(cat.Dfar) & np.isnan(cat.Dnear) & (np.abs(cat.l-180)>10)# & (cat.Dorigin==0)

			#ax.plot(X[outer], Y[outer],'.', markersize=0.5,color='k')
			XX=np.log10(X[reference])
			YY=np.log10(Y[reference])
			h,xe,ye=np.histogram2d(XX, YY,\
				bins=50, range=[[np.floor(XX.min()),np.ceil(XX.max())],[np.floor(YY.min()),np.ceil(YY.max())]])
			xe=10**(xe[:-1]/2+xe[1:]/2)
			ye=10**(ye[:-1]/2+ye[1:]/2)
			ax.contour(xe, ye, h.T, levels=np.arange(3,h.max()+10,30), colors='k', zorder=10, linewidths=0.8)
			x=np.array([1e-2,1e4])
			#ax.plot(x, 0.48*x**0.63,'--',color='limegreen')
			#ax.plot(x, 0.7778*x**0.43,'--',color='limegreen')
			ax.plot(x, 0.23*x**0.43,'--',color='limegreen')
			#ax.plot(x, 0.23*x**0.4,'--',color='green')
			

		#highlight = (self.cat.Rgal<8.15) & (self.cat.Rgal>7.5) & (self.cat.SurfaceDensity<8) & (self.cat.D>8)
		cat = Catalogue().open('clump_self0.10_equalusenear.cat')
		figname = 'fig_larson_distance_fromSRvd'
		cat.physz = cat.angsz * cat.D *1e3
		tmp(ax[0], cat, cat.Dnear)
		tmp(ax[1], cat, cat.D)
		tmp(ax[2], cat, cat.Dfar)
		#tmp(ax[1], cat.sz/2, cat.avm2, cat.Rgal, highlight)
		#tmp(ax[2], catfar.sz/2, catfar.avm2, catfar.Rgal, highlight)

		ax[1].plot([0,1],[100,100],'.',color='orangered',label='$R_{gal}$<R$_0$, use near')
		ax[1].plot([0,1],[100,100],'.',color='royalblue',label='$R_{gal}$<R$_0$, use far')
		ax[1].plot([0,1],[100,100],'-',color='k',linewidth=0.8,label='$R_{gal}$>R$_0$')
		#ax[1].plot([0,1],[100,100],'--',color='limegreen',label='$\sigma_v$=0.23 ($\Sigma$ R)$^{0.43}$')
		ax[1].plot([0,1],[100,100],'--',color='limegreen',label='$\sigma_v$=0.78 $\Sigma$R$^{0.43}$')
		ax[1].legend()

		ax[0].set_title('All use near')
		ax[1].set_title('Adopted')
		ax[2].set_title('All use far')
		for a in ax: a.set_xlabel('$\Sigma$ R (M$_\odot$ pc$^{-1}$)')
		ax[0].set_ylabel('$\sigma_v$ (km s$^{-1}$)')
		ax[2].set_xscale('log')
		ax[2].set_yscale('log')
		ax[2].set_xlim([1e-3,1.5e4])
		ax[2].set_ylim([8e-2,6])
		#ax[2].set_xlim([1e-3,60])
		#ax[2].set_ylim([8e-2,6])

		plt.show()
		#plt.savefig('%s.png' % figname, dpi=self.dpi)
		#print('Export to %s.png\n' % figname)		

	def plot_point3D(self):

		Rgal = np.load('Rgal.npy')
		D = np.load('D.npy')
		Dnear =  np.load('Dnear.npy')
		Dfar =  np.load('Dfar.npy')
		Dorigin =  np.load('Dorigin.npy')

		R = Parameter(np.load('R.npy'), unit=u.pc, \
			bins=10**np.linspace(-2.1, 1.6, 101), label=r'R')
		vd = Parameter(np.load('avm2.npy'), unit=u.km/u.s, \
			bins=10**np.linspace(-1.1, 0.7, 101), label=r'$\sigma_v$')
		SD = Parameter(np.load('SD.npy'), unit=u.Msun/u.pc**2, \
			bins=10**np.linspace(-0.3, 2.6, 101), label=r'$\Sigma$')
		SDxR = SD*R
		SDxR.bins, SDxR.label = 10**np.linspace(-3.0, 4.1, 101), r'$\Sigma$R'

		R0 = 8.15
		original = (Dorigin == 0) & np.isfinite(D)
		outer = (Rgal>R0) & original
		inner = (Rgal<R0) & original
		usenear = (D==Dnear) & inner
		usefar = (D==Dfar) & inner

		print(usenear.sum(),usefar.sum())
		def _dot_contour(ax, X, Y, **kws):
			#dots
			ax.plot(X.value[usenear], Y.value[usenear],'.',markersize=0.5,color='orangered')
			ax.plot(X.value[usefar], Y.value[usefar],'.',markersize=0.6,color='royalblue',alpha=0.6)

			#ax.plot((X/D*Dnear).value[usefar], Y.value[usefar],'.',markersize=0.5,color='k',alpha=0.6)

			#contours
			h,xe,ye=np.histogram2d(X[outer].value, Y[outer].value, bins=[X.bins, Y.bins])
			import scipy.ndimage as ndimage
			h = ndimage.gaussian_filter(h, sigma=(1,1), order=0)
			xc = np.sqrt(xe[:-1] * xe[1:])
			yc = np.sqrt(ye[:-1] * ye[1:])
			ax.contour(xc, yc, h.T, levels=np.linspace(3,h.max()*0.9,4), colors='k', zorder=10, linewidths=0.8)

			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlim(X.bins[0], X.bins[-1])
			ax.set_ylim(Y.bins[0], Y.bins[-1])

			ax.set_xlabel(X.axislabel, fontsize=font['MEDIUM'])
			ax.set_ylabel(Y.axislabel, fontsize=font['MEDIUM'])
			default = dict(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
			default.update(kws)
			ax.tick_params(**default)
			if default['labeltop']: ax.xaxis.set_label_position('top')
			if default['labelright']: ax.yaxis.set_label_position('right')

		fig,ax=plt.subplots(nrows=2, ncols=2,figsize=[6,6])
		plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,hspace=0,wspace=0)

		
		_dot_contour(ax[0,0], R, SD)
		_dot_contour(ax[1,0], R, vd)
		_dot_contour(ax[1,1], SD, vd, labelleft=False, labelright=True)
		ax[0,1].axis('off')
		'''
		_dot_contour(ax[0,0], R, vd, labeltop=True, labelbottom=False)
		_dot_contour(ax[0,1], SD, vd, labeltop=True, labelbottom=False, labelleft=False, labelright=True)
		_dot_contour(ax[1,0], R, SD)
		#newax = fig.add_axes([0.53, 0.1, 0.37, 0.37])
		_dot_contour(ax[1,1], SDxR, vd, labelright=True, labelleft=False)
		'''

		
		ax[0,1].plot([0,1],[100,100],'.',color='orangered',label='$R_{gal}$<R$_0$, use near distances')
		ax[0,1].plot([0,1],[100,100],'.',color='royalblue',label='$R_{gal}$<R$_0$, use far distances')
		ax[0,1].plot([0,1],[100,100],'-',color='k',linewidth=0.8,label='$R_{gal}$>R$_0$')
		#ax[0,1].plot([0,1],[100,100],'--',color='limegreen',label='$\sigma_v$=0.23 ($\Sigma$ R)$^{0.43}$')
		#ax[0,1].plot([0,1],[100,100],'--',color='limegreen',label='$\sigma_v$=0.78 R$^{0.43}$')
		ax[0,1].set_xlim(0,1)
		ax[0,1].set_ylim(0,1)
		ax[0,1].legend(loc='lower left', fontsize=font['SMALL'])

		#plt.show()
		figname = 'fig_point3D'
		plt.savefig('%s.png' % figname, dpi=self.dpi)
		print('Export to %s.png\n' % figname)		


	##################################################################################################################
	def _point(figname=None, dpi=600, layers=[], **kws):
		#plot point figure
		fig,ax=plt.subplots(figsize=[6,6])
		#plt.subplots_adjust(left=0.15,right=0.9,top=0.95,bottom=0.08)
		colorbar = any(['colorbar' in l for l in layers])

		for l in layers:
			method = eval('ax.'+l['method'])
			if 'colorbar' in l:
				im = method(*l['args'], **l['kws'])
				cbkws = l['colorbar']
			else:
				method(*l['args'], **l['kws'])

		ax.tick_params(axis='both', labelsize=font['SMALL'])
		ax.set_xscale('log')
		ax.set_yscale('log')
		if 'xlim' in kws: ax.set_xlim(kws['xlim'])
		if 'ylim' in kws: ax.set_ylim(kws['ylim'])
		if 'xlabel' in kws: ax.set_xlabel(kws['xlabel'],fontsize=font['MEDIUM'])
		if 'ylabel' in kws: ax.set_ylabel(kws['ylabel'],fontsize=font['MEDIUM'])

		if colorbar:
			cb_ax = fig.add_axes([1-0.75/figsize[0], figadjust['bottom'], 0.01, figadjust['top']-figadjust['bottom']])
			cbar = fig.colorbar(im, cax=cb_ax)
			#cb_ax.text(1, 1., cbkws['label'], va='bottom', ha='right', transform=cb_ax.transAxes, fontsize=font['SMALL'])
			cbar.set_label(**cbkws, fontsize=font['MEDIUM'])
			cbar.ax.tick_params(axis='both', labelsize=font['SMALL'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_point(self, n=[0,]):
		D = np.load('D.npy')
		l = np.load('l.npy')
		mask = (D>1e-2) & ((l<170) | (l>183))

		#cat=Catalogue().open('clump_self0.10_equalusenear.cat')
		R = dict(data=np.load('R.npy'), bins=10.0**np.arange(-2.2, 1.7, 0.05), label='R [pc]')
		avm2 = dict(data=np.load('avm2.npy'), bins=10.0**np.arange(-1., 0.7, 0.025), label='$\sigma_v$ (km s$^{-1}$)')
		SD = dict(data=np.load('SD.npy'), bins=10.0**np.arange(-0.8, 3.4, 0.025), label='$\Sigma$ (M$_\odot$ pc$^{-2}$)')
		mass = dict(data=np.load('mass.npy'), bins=10.0**np.arange(-3, 5.8, 0.1), label='mass (M$_\odot$)')

		if 0 in n:
			X, Y = R, mass
			layers=[]
			figname = 'fig_point_Rvd'
			#h,xe,ye = np.hist2d()
			layers.append({'method':'hist2d', 'args':(X['data'][mask], Y['data'][mask]), \
				'kws':dict(bins=[X['bins'],Y['bins']], cmap='Blues', alpha=0.8, zorder=1)})
			layers.append({'method':'plot', 'args':(X['data'][mask], Y['data'][mask], 'k.'), \
				'kws':dict(markersize=0.1,zorder=0)})
			PlotCatalogue._point(figname=figname, dpi=self.dpi, layers=layers, \
				xlim = X['bins'][[0,-2]], xlabel = X['label'], \
				ylim = Y['bins'][[0,-2]], ylabel = Y['label'])

		if 1 in n:
			l=np.load('l.npy')
			b=np.load('b.npy')
			D=np.load('D.npy')
			x=D*np.cos(b*dtor)*np.cos((l-90)*dtor)
			y=D*np.cos(b*dtor)*np.sin((l-90)*dtor)+8.15

			grd=np.load('gradient.npy')
			gx=grd[:,0]
			gy=grd[:,1]
			g = np.sqrt(gx**2+gy**2)
			ang = np.arctan(gy/gx)/np.pi*180

			fig,ax=plt.subplots()
			#ax.hist(g,bins=10**np.linspace(-3.2,3,50))
			ax.hist(ang,bins=10**np.linspace(-3.2,3,50))
			#ax.scatter(x,y,c=g/D,norm=LogNorm(),vmin=1e-1,vmax=2,s=0.2,cmap='rainbow')
			#ax.scatter(x,y,c=np.abs(ang),vmin=0,vmax=90,s=0.1,cmap='rainbow')
			#ax.set_aspect('equal')
			#ax.plot(D,g/D,'.',markersize=0.2)
			#ax.set_xscale('log')
			#ax.set_yscale('log')
			#for r in np.arange(-5,5,1):
			#	idx=(b>r) & (b<r+1)
			#	h,ex = np.histogram(ang[idx], bins=np.linspace(-90,90,6), density=True)
			#	ax.step((ex[:-1]+ex[1:])/2,h,where='mid')
			plt.show()


	##################################################################################################################
	def _hist(value, components, labels, figname=None, dpi=600, xlog=True, ylog=True, **kws):
		#plot a histogram figure
		fig, ax = plt.subplots(figsize=[7,5])

		if xlog: x = np.sqrt(value.bins[:-1] * value.bins[1:])
		else: x = (value.bins[:-1]+value.bins[1:])/2
		for c,l in zip(components,labels):
			v=value[c].value
			v=v[v>0]
			meanv = 10**np.nanmean(np.log10(v))
			hist, edge = np.histogram(v, bins=value.bins)
			ax.step(x, hist, where='mid', label='%s (%.2f)' % (l, meanv), color='k' if l=='All' else None)
		ax.legend(fontsize=font['MEDIUM'])

		ax.tick_params(axis='both', labelsize=font['SMALL'])
		if 'xlabel' in kws: ax.set_xlabel(kws['xlabel'], fontsize=font['MEDIUM'])
		else: ax.set_xlabel(value.axislabel, fontsize=font['MEDIUM'])
		if 'ylabel' in kws: ax.set_ylabel(kws['ylabel'], fontsize=font['MEDIUM'])
		else: ax.set_ylabel('N', fontsize=font['MEDIUM'])
		if xlog: ax.set_xscale('log')
		if ylog: ax.set_yscale('log')

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_hist(self, n=range(8)):
		#cat=Catalogue().open('clump_self0.10.cat')
		D = np.load('D.npy')
		l = np.load('l.npy')
		mask = (D>1e-2) & ((l<170) | (l>183))

		if 1:
			suffix='D'
			sep = [0,1,4,9,16,25]#range(0,21,4)#
			components = [(sep[i]<D) & (D<sep[i+1]) & mask for i in range(len(sep)-1)]
			labels = [('%2i<' % sep[i] if i>0 else '')+'D<%2i kpc' % sep[i+1] for i in range(len(sep)-1)]
		else:
			suffix='Rgal'
			Rgal = np.load('Rgal.npy')
			sep = [1,4,9,16,25]#range(0,21,4)#
			components = [(sep[i]<Rgal) & (Rgal<sep[i+1]) & mask for i in range(len(sep)-1)]
			labels = ['%2i < R$_{gal}$ < %2i kpc' % (sep[i],sep[i+1]) for i in range(len(sep)-1)]
		components.append(mask)
		labels.append('All')

		if 0 in n:
			figname = 'fig_hist_SD_%s' % suffix
			SD = Parameter(np.load('SD.npy'), unit=u.Msun/u.pc**2, \
				bins=10.0**np.arange(-0.8, 3.4, 0.05), label=r'$\Sigma$')
			print(SD[mask].min(),SD[mask].max())
			PlotCatalogue._hist(SD, components, labels, figname=figname)

		if 1 in n:
			figname = 'fig_hist_mass_%s' % suffix
			mass = Parameter(np.load('mass.npy'), unit=u.Msun, \
				bins=10.0**np.arange(-5.5, 5.8, 0.2), label=r'M$_{LTE}$')
			print(mass[mask].min(),mass[mask].max())
			PlotCatalogue._hist(mass, components, labels, figname=figname)

		if 2 in n:
			figname = 'fig_hist_R_%s' % suffix
			R = Parameter(np.load('R.npy'), unit=u.pc, \
				bins = 10.0**np.arange(-3.2, 1.7, 0.1), label=r'R')
			print(np.nanmin(R[mask]),np.nanmax(R[mask]))			
			PlotCatalogue._hist(R, components, labels, figname=figname)

		if 3 in n:
			figname = 'fig_hist_n_%s' % suffix
			value = np.load('n.npy')
			vdensity = Parameter(np.load('n.npy'), unit=u.cm**-3, \
				bins = 10.0**np.arange(-1, 6, 0.1), label=r'n')
			print(np.nanmin(vdensity[mask]),np.nanmax(vdensity[mask]))
			PlotCatalogue._hist(vdensity, components, labels, figname=figname)

		if 4 in n:
			figname = 'fig_hist_vd_%s' % suffix
			vd = Parameter(np.load('avm2.npy'), unit=u.km/u.s, \
				bins = 10.0**np.arange(-1.9, 0.8, 0.05), label=r'$\sigma_v$')
			print(np.nanmin(vd[mask]),np.nanmax(vd[mask]))
			PlotCatalogue._hist(vd, components, labels, figname=figname)

		if 5 in n:
			figname = 'fig_hist_ntvd_%s' % suffix
			avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, \
				bins=10.0**np.arange(-1.9, 0.8, 0.05), label=r'$\sigma_{nt}$')
			print(np.nanmin(avntvd[mask]),np.nanmax(avntvd[mask]))
			PlotCatalogue._hist(avntvd, components, labels, figname=figname)

		if 6 in n:
			figname = 'fig_hist_Tex_%s' % suffix
			Tex = Parameter(np.load('Tex.npy'), unit=u.K, \
				bins=10.0**np.arange(0, 2, 0.05), label=r'T$_{ex}$')
			print(np.nanmin(Tex[mask]),np.nanmax(Tex[mask]))
			PlotCatalogue._hist(Tex, components, labels, figname=figname)

		if 7 in n:
			figname = 'fig_hist_grd_%s' % suffix
			gradient = Parameter(np.sqrt(np.sum(np.load('gradient.npy')**2,axis=1)), unit=u.km/u.s/u.pc, \
				bins=10**np.linspace(-3.2,3,40), label='Gradient')
			print(np.nanmin(gradient[mask]),np.nanmax(gradient[mask]))
			PlotCatalogue._hist(gradient, components, labels, figname=figname, xlog=True)


	##################################################################################################################
	def _hist2d(X={}, Y={}, C=None, lines=[], figname=None, dpi=600, fitting=True, **kws):
		#plot a 2d histogram figure

		def view(var):
			v=np.log10(var.value)
			v=v[np.isfinite(v)]
			print(var.label)
			print('min=%+.2f\t1=%+.2f\tmean%+.2f\t99=%+.2f\tmax=%+.2f' % \
			(v.min(), np.percentile(v,1), v.mean(), np.percentile(v,99), v.max()))
		view(X)
		view(Y)

		fig,ax=plt.subplots(figsize=[6,6])
		#plt.subplots_adjust(left=0.15,right=0.9,top=0.95,bottom=0.08)
		if C is None:
			im=ax.hist2d(X.value, Y.value, bins=[X.bins, Y.bins], cmap='viridis', zorder=1, alpha=0.9, norm=LogNorm())
			#gist_heat_r
			#CMRmap_r
			#nipy_spectral_r
		else:
			im=ax.scatter(X, Y, c=C,cmap='rainbow',s=0.2,zorder=0, norm=LogNorm())

		if fitting:
			logX=np.log10(X.value)
			if X.error is not None: logXerr=X.error/X/np.log(10)
			else: logXerr=np.zeros(logX.shape)
			logY=np.log10(Y.value)
			if Y.error is not None: logYerr=Y.error/Y/np.log(10)
			else: logYerr=np.zeros(logY.shape)
			mask = np.isfinite(logX) & np.isfinite(logY)# & (logX>-1)
			
			import bces.bces as BCES
			a,b,aerr,berr,covab = BCES.bcesp(logX[mask], logXerr[mask], logY[mask], logYerr[mask], np.zeros(mask.sum()))
			print(a,b,aerr,berr,covab)
			usemethod=2
			lines.insert(0, (10**b[usemethod], a[usemethod], 'k-'))

		#other lines
		def loglinear(x, A, s):
			return [A*v**s for v in x]
		positivecorrelation = True
		for l in lines:
			#l = [factor, exponent, lineformat, xrange, label]
			x = l[3] if len(l)>=4 else None
			if x is None: x=[1e-10,1e10]
			if len(l)>=5: label = l[4]
			else:
				label = '%s=%.2f' % (Y.label, l[0])
				if l[1]!=0: label += ' %s' % X.label
				if l[1]!=0 and l[1]!=1: label += '$^{%.2f}$' % l[1]
			plt.plot(x, loglinear(x,l[0],l[1]), l[2], label=label)
			if l[1]<0: positivecorrelation=False
		if positivecorrelation: plt.legend(loc='lower right')
		else: plt.legend(loc='lower left')

		#ax.tick_params(axis='both', labelsize=font['SMALL'])
		ax.set_xscale('log')
		ax.set_yscale('log')
		def axislim(par):
			if par.bins is None: return [np.nanmin(par), np.nanmax(par)]
			else: return par.bins[[0,-1]]
		ax.set_xlim(axislim(X))
		ax.set_ylim(axislim(Y))
		ax.set_xlabel(X.axislabel, fontsize=font['MEDIUM'])
		ax.set_ylabel(Y.axislabel, fontsize=font['MEDIUM'])
		
		#colorbar
		if isinstance(im,tuple): im=im[3]
		if positivecorrelation:
			axin = inset_axes(ax, width='40%', height='3%', loc='upper left', \
				bbox_to_anchor=(0.05,0,1,0.97), bbox_transform=ax.transAxes)
		else:
			axin = inset_axes(ax, width='40%', height='3%', loc='upper right', \
				bbox_to_anchor=(-0.05,0,1,0.97), bbox_transform=ax.transAxes)
		cb = fig.colorbar(im, cax=axin, orientation='horizontal', pad=0.05)
		cb.set_label('N' if C is None else C.label, fontsize=font['SMALL'])
		cb.ax.tick_params(axis='both', labelsize=font['SMALL'])
		
		if isinstance(figname, str):
			plt.savefig('%s.png' % figname)#, dpi=dpi)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_hist2d(self, n=(11,)):

		D = Parameter(np.load('D.npy'), unit=u.kpc, \
			label='Distance')
		l = Parameter(np.load('l.npy'), unit=u.deg, \
			label='Galactic Longitude')

		#the size of half maximum
		R = Parameter(np.load('R.npy'), unit=u.pc, resolution=52/3600*dtor * D*1e3, \
			bins=10**np.linspace(-2.4, 1.7, 200), label=r'R')
		vd = Parameter(np.load('avm2.npy'), unit=u.km/u.s, resolution=0.168/2.355, \
			bins=10**np.linspace(-1.2, 0.8, 200), label=r'$\sigma_v$')
		angsz = Parameter(np.load('angsz.npy'), unit=u.rad, \
			label = 'Angular Size')
		#the size of lowest contour
		area = Parameter(np.load('area.npy'), label=r'Area (pixel)')
		Rboundary = Parameter(np.sqrt(np.load('area.npy')/np.pi)*30, unit=u.arcsec, resolution=52/3600*dtor * D *1e3, \
			bins=10**np.linspace(-2.4, 1.7, 200), label=r'R')
		#Rboundary = Parameter(np.sqrt(np.load('area.npy')/np.pi)*30/3600/180*np.pi*D.to('pc').value, unit=u.pc, resolution=52/3600*dtor * D *1e3, \
		#	bins=10**np.linspace(-2.4, 1.7, 200), label=r'R [pc]')

		SD = Parameter(np.load('SD.npy'), unit=u.Msun/u.pc**2, \
			bins=10**np.linspace(-0.4, 2.7, 200), label=r'$\Sigma$')
		mass = Parameter(np.load('mass.npy'), unit=u.Msun, \
			bins=10**np.linspace(-3.0, 5.8, 200), label=r'M$_{LTE}$')
		Tex = Parameter(np.load('Tex.npy'), unit=u.K, \
			bins=10**np.linspace(-0.1, 2.0, 200), label=r'T$_{ex}$')
		vdensity = Parameter(np.load('n.npy'), unit=u.cm**-3, \
			bins=10**np.linspace(0.5, 5.3, 150), label=r'n')

		tvd = Parameter(np.load('tvd.npy'), unit=u.km/u.s, \
			label=r'$\sigma_{th}$')
		ntvd = Parameter(np.load('ntvd.npy')[:,1], unit=u.km/u.s, \
			bins=10**np.linspace(-1,0.8,200), label=r'Averaged $\sigma_{nt}$ of spectra')
		avntvd = Parameter(np.load('avntvd.npy'), unit=u.km/u.s, \
			bins=10**np.linspace(-1,0.8,200), label=r'$\sigma_{nt}$ of averaged spectrum')

		SDxR = SD*R
		SDxR.bins, SDxR.label = 10**np.linspace(-3.0, 4.1, 200), r'$\Sigma$R'
		Mach = avntvd/tvd
		Mach.bins, Mach.label = 10**np.linspace(-0.4, 1.2, 200), r'Mach Number'
		vddR2 = vd/R**0.5
		vddR2.bins, vddR2.label = 10**np.linspace(-1.5, 1.2, 200), r'$\sigma_v$/R$^{0.5}$'

		MLTE=mass
		Mvir = Parameter(1162.5*R.value*vd.value**2, unit=u.Msun, \
			bins=10**np.linspace(-3,5.8,200), label=r'M$_{vir}$')#-1.0, 5.0, 
		MJeans = Parameter(17.3*Tex.value**1.5*vdensity.value**(-0.5), unit=u.Msun, \
			bins=10**np.linspace(-3,5.8, 200), label=r'M$_{Jeans}$')#-0.0, 3.0
		alpha = Mvir/MLTE
		alpha.bins, alpha.label = 10**np.linspace(-1.0, 3.0, 200), r'$\alpha_{vir}$'

		good = (D>5e-2*u.kpc) & ((l<170*u.deg) | (l>183*u.deg))

		_C = Parameter(np.ones(D.shape))

		if 0 in n:
			figname='fig_hist2d_R_vd'
			PlotCatalogue._hist2d(R[good], vd[good], figname=figname, lines=[(0.48,0.63,'r--')])
			figname='fig_hist2d_SD_vd'
			PlotCatalogue._hist2d(SD[good], vd[good], figname=figname)
		if 1 in n:
			figname='fig_hist2d_R_mass'
			line = [(v*np.pi,2,':',None,r'$\Sigma$ = %i %s' % (v,u.Unit('Msun/pc2').to_string('latex_inline'))) for v in [10,100]]
			PlotCatalogue._hist2d(R[good], mass[good], figname=figname, lines=line+[(36.7,2.2,'r--')])
		if 2 in n:
			figname='fig_hist2d_mass_vd'
			PlotCatalogue._hist2d(mass[good], vd[good], figname=figname, lines=[(0.19,0.27,'r--')])
		if 3 in n:
			figname='fig_hist2d_SDxR_vd'
			PlotCatalogue._hist2d(SDxR[good], vd[good], figname=figname, lines=[(0.23,0.43,'r--')])
		if 4 in n:
			figname='fig_hist2d_SD_vddR2'
			PlotCatalogue._hist2d(SD[good], vddR2[good], figname=figname, lines=[(0.09,0.5,'g:',None,r'$\alpha_{vir}$ = 3')])

		if 5 in n:
			figname='fig_hist2d_MLTE_Mvir'
			PlotCatalogue._hist2d(mass[good], Mvir[good], figname=figname, fitting=False, \
				lines=[(3,1,'g:',None,r'$\alpha_{vir}$ = 3'), (1,1,'b:',None,r'$\alpha_{vir}$ = 1')])
		if 6 in n:
			figname='fig_hist2d_MLET_MJeans'
			PlotCatalogue._hist2d(mass[good], MJeans[good], figname=figname, fitting=False, \
				lines=[(1,1,'g:',None,r'M$_{LTE}$ = M$_{Jeans}$')])
		if 7 in n:
			figname='fig_hist2d_MLET_alpha'
			PlotCatalogue._hist2d(mass[good], alpha[good], figname=figname, \
				lines=[(3,0,'g:',None,r'$\alpha_{vir}$ = 3'), (1,0,'b:',None,r'$\alpha_{vir}$ = 1')])
		if 8 in n:
			figname='fig_hist2d_R_Mach'
			PlotCatalogue._hist2d(R[good], Mach[good], figname=figname, \
				lines=[(1,0,'g:',None,r'Mach = 1'), (2,0,'b:',None,r'Mach = 2')])
		if 9 in n:
			figname='fig_hist2d_SDxR_Mach'
			PlotCatalogue._hist2d(SDxR[good], Mach[good], figname=figname, \
				lines=[(1,0,'g:',None,r'Mach = 1'), (2,0,'b:',None,r'Mach = 2')])
		if 10 in n:
			figname='fig_hist2d_avntvd_ntvd'
			PlotCatalogue._hist2d(ntvd[good], avntvd[good], figname=figname, fitting=False, lines=[(1,1,'g:')])


		Q=1#0.5
		rho = vdensity*2.83*con.m_p
		S = Parameter(np.load('BfieldS.npy'), unit=u.deg, \
			bins=10.0**np.linspace(-0.5, 1.8, 200), label=r'$\sigma_\theta$')
		Bpos_classic = Parameter((np.sqrt(4*np.pi*rho) * avntvd / S * Q).cgs.value*1e6, unit=u.uG, \
			bins=10**np.linspace(-0.3,3.9,150), label=r'B$_{POS}$')
		Bpos_alter = Parameter((np.sqrt(2*np.pi*rho) * avntvd / S**0.5 * Q).cgs.value*1e6, unit=u.uG, \
			bins=10**np.linspace(-0.3,3.9,150), label=r'B$_{POS}$')
		gradient = Parameter(np.sqrt(np.sum(np.load('gradient.npy')**2,axis=1)), unit=u.km/u.s/u.pc, \
			bins=10**np.linspace(-2,2,200), label='Gradient')

		myvalue = avntvd/S
		myvalue.bins = np.logspace(-3,1,200)
		myvalue.label = 'MyValue'
		print(myvalue.lim)

		#ncomp = np.load('ncomp2kpc.npy')
		near = good & (D<2*u.kpc) & (Rboundary>3.5*u.arcmin) & (S<25*u.deg)# & (ncomp<3)
		if 11 in n:
			figname=None#'fig_hist2d_n_Bposclassic'
			PlotCatalogue._hist2d(vdensity[near], Bpos_classic[near], figname=figname, fitting=False, \
				lines=[(10/300**0.65,0.65,'k--',[300,1e10],'Cruther et al. 2010'), (10,0,'k--',[1e-10,300],None)])
			#figname='fig_hist2d_n_Bposalter'
			#PlotCatalogue._hist2d(vdensity[near], Bpos_alter[near], figname=figname, fitting=False, \
			#	lines=[(10/300**0.65,0.65,'k--',[300,1e10],'Crutcher et al. 2010'), (10,0,'k--',[1e-10,300],None)])
			#PlotCatalogue._hist2d(vdensity[good], R[good], figname=figname, fitting=False, \
			#	lines=[(10/300**0.65,0.65,'k--',[300,1e10],'Crutcher et al. 2010'), (10,0,'k--',[1e-10,300],None)])
		if 12 in n:
			figname=None
			PlotCatalogue._hist2d(S[near], gradient[near], C=vdensity[near], figname=figname, fitting=False, \
				lines=[(10/300**0.65,0.65,'k--',[300,1e10],'Crutcher et al. 2010'), (10,0,'k--',[1e-10,300],None)])
		if 13 in n:
			figname=None
			PlotCatalogue._hist2d(SD[good], avntvd[good], figname=figname, fitting=False)

	##################################################################################################################

	def _binmean(*args, threshold=1, figname=None, dpi=600, nlimit=5):
		#open fits to data
		def dict2Parameter(d):
			value = fits.open(d['datafile'])[0].data.ravel()
			if 'errorfile' in d: error = fits.open(d['errorfile'])[0].data.ravel()
			else: error = np.ones(value.shape)
			if 'unit' not in d: d['unit'] = u.Unit()
			if 'meanbins' not in d: d['meanbins'] = None
			return Parameter(value, unit=d['unit'], error=error, \
				label=d['label'], bins=d['bins'], meanbins=d['meanbins'], scale=d['scale'])
		X = dict2Parameter(args[0])
		Y = dict2Parameter(args[1])
		if len(args)==3:
			ncomp = dict2Parameter(args[2])
		print('X ranges from %e to %e %s' % (*X.lim[[0,2]].value, X.unit.to_string()))
		print('Y ranges from %e to %e %s' % (*Y.lim[[0,2]].value, Y.unit.to_string()))

		fig,ax = plt.subplots(figsize=(7,6))
		if X.scale is not None: ax.set_xscale(X.scale)
		if Y.scale is not None: ax.set_yscale(Y.scale)

		if len(args)==3:
			#im=ax.plot(X.ravel(), Y.ravel(),'k,', alpha=1)
			#h,xe,ye = np.histogram2d(X.value.ravel(), Y.value.ravel(), bins=[X.bins, Y.bins])
			#print(h.max(),h.min())
			ax.hist2d(X.value.ravel(), Y.value.ravel(), bins=[X.bins, Y.bins], \
				cmap='gray_r', zorder=1, alpha=1, norm=LogNorm(10,500))#h.max()/10,h.max()))

			mycmap=cm.get_cmap('rainbow_r',nlimit)
			
			#Contours
			import scipy.ndimage as ndimage
			ncomp[ncomp>nlimit]=nlimit
			mycmap=cm.get_cmap('rainbow_r',nlimit)
			for i in range(1,nlimit+1):
				print('Ncomp=%i, %i pixels' % (i,(ncomp==i).sum()))
				h,xe,ye = np.histogram2d(X[ncomp==i].value, Y[ncomp==i].value, bins=[X.bins[::1], Y.bins[::1]])
				xc = (xe[:-1]*xe[1:])**0.5 if X.scale=='log' else (xe[:-1]+xe[1:])/2
				yc = (ye[:-1]*ye[1:])**0.5 if Y.scale=='log' else (ye[:-1]+ye[1:])/2
				h = ndimage.gaussian_filter(h, sigma=(3,5), order=0)
				ax.contour(xc, yc, h.T, levels=[h.max()*0.5,h.max()*0.7, h.max()*0.9], \
					colors=[mycmap(i-1),], zorder=10-i, linewidths=1., alpha=0.7)
			
			#for colorbar, draw hidden scatter
			im = ax.scatter([1e-5]*nlimit, [1e-5]*nlimit, c=[1]*nlimit, cmap=mycmap, vmin=0.5, vmax=nlimit+0.5)

			#draw bin means
			def bin_XY(x, y, yerr=1, bins=None, xscale='log', yscale='log'):
				bin_edges = np.linspace(np.nanmin(x), np.nanmax(x), 10) if bins is None else bins
				if isinstance(yerr, (int,float)): yerr = np.repeat(yerr, y.size)
				bin_mean = []
				bin_std = []
				bin_median = []
				bin_pct4 = []
				goodY = np.isfinite(y) & np.isfinite(yerr)
				for e0,e1 in zip(bin_edges[:-1],bin_edges[1:]):
					idx = (x>=e0) & (x<e1) & goodY
					print('In bin (%.2e, %.2e) %i pixels' % (e0,e1,idx.sum()))
					if idx.sum()>2:
						values = y[idx]
						median = np.median(values)
						pct4 = np.abs(np.percentile(values, [25,75]) - median)
						if yscale=='log':
							weights = 1/(yerr[idx]/values)**2#1/yerr[idx]**2
							values = np.log10(values)
						else:
							weights = 1/yerr[idx]**2
						mean = np.average(values, weights=weights)
						std = np.sqrt(np.average((values-mean)**2, weights=weights))
						if yscale=='log':
							std = [10**mean-10**(mean-std),10**(mean+std)-10**mean]
							mean = 10**mean
						else:
							std = [std,std]
					else:
						mean, std = np.nan, [np.nan,np.nan]
						median, pct4 = np.nan, [np.nan,np.nan]

					#print(mean,std, median, pct4)
					bin_mean.append(mean)
					bin_std.append(std)
					bin_median.append(median)
					bin_pct4.append(pct4)
				bin_center = (bin_edges[:-1]*bin_edges[1:])**0.5 if xscale=='log' else (bin_edges[:-1]+bin_edges[1:])/2
				return bin_center, np.array(bin_median), np.array(bin_pct4)

			def weighted_percentile(data, weights, perc):
				#perc : percentile in [0-1]!
				ix = np.argsort(data)
				data = data[ix] # sort data
				weights = weights[ix] # sort weights
				cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
				return np.interp(perc, cdf, data)

			def linear_running_mean(x, y, yerr=None, n=1000):
				#x,y should NOT contain nan, should be in LINEAR scale
				if yerr is None: yerr = np.ones_like(y)
				good = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
				print('There are %i points' % good.sum())
				x, y, yerr = x[good], y[good], yerr[good]
				idx = np.argsort(x)
				weights = 1/yerr**2
				bins, means, errors = [], [], [[],[]]
				start = idx.size%n // 2 #drop the first and last several
				for i in range(start, idx.size-n, n):
					subidx = idx[i:i+n]
					bins.append(np.mean(x[subidx]))
					lo,cen,up = weighted_percentile(y[subidx], weights[subidx], [0.25,0.5,0.75])
					means.append(cen)
					errors[0].append(cen-lo)
					errors[1].append(up-cen)
				return np.array(bins), np.array(means), np.array(errors)

			def log_running_means(x, y, yerr=None, n=1000):
				lgx = np.log10(x)
				lgy = np.log10(y)
				lgyerr = None if yerr is None else yerr/y
				lgbins, lgmeans, lgerrors = linear_running_mean(lgx, lgy, yerr=lgyerr, n=n)
				bins = 10**lgbins
				means = 10**lgmeans
				errors = np.array([means-10**(lgmeans-lgerrors[0]), 10**(lgmeans+lgerrors[1])-means])
				return bins, means, errors

			#bin_center, bin_mean, bin_std = bin_XY(X[ncomp==1].value, Y[ncomp==1].value, yerr=Y.error[ncomp==1], \
			#	bins=X.meanbins, xscale=X.scale, yscale=Y.scale)
			bin_center, bin_mean, bin_err = log_running_means(X[ncomp==1].value, Y[ncomp==1].value, yerr=None, n=150000)#Y.error[ncomp==1]
			ax.errorbar(bin_center, bin_mean, yerr=bin_err, linestyle='None', linewidth=1.5, \
				fmt='.', color=mycmap(0), capsize=3, zorder=12)
			#ax.fill_between(bin_center, bin_mean-bin_std[:,0], bin_mean+bin_std[:,1], color=mycmap(0),edgecolor=None,alpha=0.2)
			#ax.plot(bin_center, bin_mean, color=mycmap(0))

			#colorbar
			axin = inset_axes(ax, width='25%', height='3%', loc='upper left', \
				bbox_to_anchor=(0.05,0,1,0.97), bbox_transform=ax.transAxes)
			cb = fig.colorbar(im, cax=axin, orientation='horizontal', pad=0.05, ticks=np.arange(1,nlimit+1))
			xticklabels = [str(n) for n in range(1,nlimit+1)]
			xticklabels[-1] = r'$\geq$'+xticklabels[-1]
			cb.ax.set_xticklabels(xticklabels)
			cb.ax.tick_params(axis='both', labelsize=font['SMALL'])
			cb.set_label('N$_{component}$', fontsize=font['SMALL'])
			cb.outline.set_visible(False)
			ax.set_xlim(X.bins[[0,-1]])
			ax.set_ylim(Y.bins[[0,-1]])
		else:
			###ViolinPlot for Ncomp
			ncomp=X
			ncomp[ncomp>nlimit]=nlimit
			data=[]
			posi=np.arange(1,nlimit+1)
			for n in range(1,nlimit+1):
				idx = ncomp.value == n
				if idx.sum()>2:
					data.append(Y[idx].value)
			parts = ax.violinplot(data, positions=posi, widths=0.8, showmedians=True)
			for partname in ('cbars','cmins','cmaxes','cmedians'):
				vp = parts[partname]
				vp.set_edgecolor('black')
				vp.set_linewidth(1)
			for pc in parts['bodies']:
				pc.set_color('black')
				pc.set_facecolor('#D43F3A')
				pc.set_edgecolor('black')
				pc.set_alpha(1)
			xticklabels = ['%i' % n for n in posi]
			xticklabels[-1] = r'$\geq$'+xticklabels[-1]
			ax.set_xticks(posi)
			ax.set_xticklabels(xticklabels)

		ax.set_xlabel(X.axislabel,fontsize = font['MEDIUM'])
		ax.set_ylabel(Y.axislabel,fontsize = font['MEDIUM'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png\n' % figname)			
		else:
			plt.show()

	def plot_binmean(self, n=range(5)):
		Dlimit = '2kpc'	#''/'1kpc'/'2kpc'/'3kpc'/'5kpc'
		fwhm = '7'	#'7'/'10'
		ns = 2048 if fwhm=='7' else 1024
		#Xs
		logntvd = dict(datafile='Bfield/nt%s_fwhm%s.fits' % (Dlimit,fwhm), unit=u.km/u.s, scale='log', 
			bins=np.logspace(np.log10(7e-2),np.log10(1.2),200), meanbins=np.logspace(np.log10(1e-1),np.log10(8e-1),11), \
			label=r'$\sigma_{nt}$')
		ntvd = dict(datafile='Bfield/nt%s_fwhm%s.fits' % (Dlimit,fwhm), unit=u.km/u.s, scale='linear', \
			bins=np.linspace(0, 1.5, 110), meanbins=np.linspace(0,1.2,12), \
			label=r'$\sigma_{nt}$')

		logN = dict(datafile='Bfield/N%s_fwhm%s.fits' % (Dlimit,fwhm), unit=u.cm**-2 ,scale='log', \
			bins=np.logspace(np.log10(3e19), np.log10(6e21), 200), meanbins=np.logspace(np.log10(8e19),np.log10(2e21),11), \
			label=r'N')

		logmach = dict(datafile='Bfield/mach%s_fwhm%s.fits' % (Dlimit,fwhm), scale='log', \
			bins=np.logspace(np.log10(0.2), np.log10(6), 200), meanbins=np.logspace(np.log10(5e-1),np.log10(4),11), \
			label=r'Mach Number')
		mach = dict(datafile='Bfield/mach%s_fwhm%s.fits' % (Dlimit,fwhm), scale='linear', \
			bins=np.linspace(0, 20, 110), meanbins=np.linspace(0,20,40), \
			label=r'Mach Number')

		ncomp = dict(datafile='ncomp%s.fits' % Dlimit, scale='linear', \
			bins=np.linspace(0, 9, 80), meanbins=np.arange(0.5, 8.6, 1.0), \
			label=r'N$_{component}$')

		#Ys
		#polarization fraction (p), dispersion in polarization angles (S)
		logSp = dict(datafile='Bfield/interp_nearest_mapSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), unit=u.deg, \
			errorfile='Bfield/interp_nearest_mapsigSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='log', \
			bins=np.logspace(np.log10(0.03), np.log10(0.4), 200), label=r'S p')
		Sp = dict(datafile='Bfield/interp_nearest_mapSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), unit=u.deg, \
			errorfile='Bfield/interp_nearest_mapsigSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='linear', \
			bins=np.linspace(0, 1, 200), label=r'S p')

		logS = dict(datafile='Bfield/interp_nearest_mapS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), unit=u.deg, \
			errorfile='Bfield/interp_nearest_mapsigS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='log', \
			bins=np.logspace(np.log10(0.6), np.log10(60), 200), label=r'S')
		S = dict(datafile='Bfield/interp_nearest_mapS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), unit=u.deg, \
			errorfile='Bfield/interp_nearest_mapsigS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='linear', \
			bins=np.linspace(0, 25, 85), label=r'S')

		logp = dict(datafile='Bfield/interp_nearest_mapp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='log', \
			bins=np.logspace(np.log10(3e-3), np.log10(0.2), 200), label=r'p')

		if 0 in n:
			figname = 'fig_binmean_ntvd_S'
			PlotCatalogue._binmean(logntvd, logS, ncomp, figname=figname)
			figname = 'fig_binmean_ntvd_Sp'
			PlotCatalogue._binmean(logntvd, logSp, ncomp, figname=figname)
			figname = 'fig_binmean_ntvd_p'
			PlotCatalogue._binmean(logntvd, logp, ncomp, figname=figname)
		if 1 in n:
			figname = 'fig_binmean_N_S'
			PlotCatalogue._binmean(logN, logS, ncomp, figname=figname)
			figname = 'fig_binmean_N_Sp'
			PlotCatalogue._binmean(logN, logSp, ncomp, figname=figname)
			figname = 'fig_binmean_N_p'
			PlotCatalogue._binmean(logN, logp, ncomp, figname=figname)
		if 2 in n:
			figname = 'fig_binmean_mach_S'
			PlotCatalogue._binmean(logmach, logS, ncomp, figname=figname)
			figname = 'fig_binmean_mach_Sp'
			PlotCatalogue._binmean(logmach, logSp, ncomp, figname=figname)
			figname = 'fig_binmean_mach_p'
			PlotCatalogue._binmean(logmach, logp, ncomp, figname=figname)
		if 3 in n:
			figname = 'fig_binmean_ncomp_S'
			PlotCatalogue._binmean(ncomp, logS, figname=figname, nlimit=8)
			figname = 'fig_binmean_ncomp_Sp'
			PlotCatalogue._binmean(ncomp, logSp, figname=figname, nlimit=8)
			figname = 'fig_binmean_ncomp_p'
			PlotCatalogue._binmean(ncomp, logp, figname=figname, nlimit=8)
		if 4 in n:
			figname = 'fig_binmean_ncomp_ntvd'
			PlotCatalogue._binmean(ncomp, logntvd, figname=figname, nlimit=8)
			figname = 'fig_binmean_ncomp_mach'
			PlotCatalogue._binmean(ncomp, logmach, figname=figname, nlimit=8)

if __name__ == '__main__':
	a=PlotCatalogue()#catalogue='label/035.208+2.042_clump_nt.cat')
	#a.plot_lbmap()
	#a.plot_gplane()
	#a.plot_point3('fig_larson_distance')
	#a.plot_point3D()
	#a.plot_point()
	#a.plot_hist()
	#a.plot_hist2d()
	a.plot_binmean()
