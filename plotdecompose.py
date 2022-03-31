import os,glob,sys,getpass
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
from matplotlib.colors import LogNorm
from matplotlib import cm
dtor=np.pi/180
font = dict(SMALL = 8, MEDIUM = 10, LARGE = 12)


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
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_lbmap(self, n=[3,]):
		#cat = Catalogue().open('clump_tnt.cat')
		layers=[]
		if 0 in n:
			figname = 'fig_lbmap_vs'
			hdu = fits.open('/Users/sz268601/Work/GuideMap/whole/tile_L_m0.fits')[0]
			img = PlotCatalogue._squareroot(hdu.data,0,18)
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,), \
				'kws':dict(origin='lower', extent=ext, cmap='gray')})

			cmap = PlotCatalogue._truncate_colormap(plt.get_cmap('gist_rainbow'), 0.8, 0.)
			layers.append({'method':'scatter', 'args':(cat.l, cat.b), \
				'kws':dict(c=cat.avm1, vmin=-50, vmax=50, cmap=cmap, s=np.sqrt(cat.sx*cat.sy)/5, \
				marker='.', edgecolors='none', alpha=0.7),\
				'colorbar':dict(label='V$_{lsr}$ (km s$^{-1}$)')})

			#idx=(cat.Rgal>6.5) & (cat.Rgal<8.5) & (np.abs(cat.D*np.sin(cat.b*np.pi/180))>0.5) & (cat.SurfaceDensity<3)
			#layers.append({'method':'plot', 'args':(cat.l[idx], cat.b[idx],'w+'), \
			#'kws':dict(markersize=1.5)})

		if 1 in n:
			figname = 'fig_lbmap_cnt1'
			hdu = fits.open('/Users/sz268601/Work/GuideMap/local/tile_L_m0.fits')[0]
			img = PlotCatalogue._squareroot(hdu.data,0,0.1)
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,), \
				'kws':dict(origin='lower', extent=ext, cmap='gray', vmin=-5, vmax=1)})

			cat = cat[np.abs(cat.avm1)<30]
			layers.append({'method':'scatter', 'args':(cat.l, cat.b), \
				'kws':dict(c=np.log10(cat.avntvd), vmin=-0.63, vmax=0, cmap='jet', s=1.5, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='log($\sigma_{nonthermal}$ [km s$^{-1}$])')})
 
		if 2 in n:
			figname = 'fig_lbmap_ntmap_whole'
			hdu = fits.open('nt.fits')[0]
			import scipy.ndimage as ndimage
			nan = np.isnan(hdu.data)
			hdu.data[nan]=0.17/2.355
			img = np.log10(hdu.data)
			#img = np.log10(ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0))
			img[nan]=-1#np.nan
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,),\
				'kws':dict(origin='lower', extent=ext, vmin=-0.63, vmax=0, cmap='jet'), \
				'colorbar':dict(label='log($\sigma_{nonthermal}$ [km s$^{-1}$])')})

		if 3 in n:
			figname = 'fig_lbmap_Spmap'
			hdu = fits.open('Bfield/interp_nearest_mapSp_fwhm7_ns2048_AngSt1.fits')[0]
			#import scipy.ndimage as ndimage
			#nan = np.isnan(hdu.data)
			#hdu.data[nan]=0.17/2.355
			img = hdu.data
			#import scipy.ndimage as ndimage
			#img = np.log10(ndimage.gaussian_filter(hdu.data, sigma=(7.8/2.355, 7.8/2.355), order=0))
			#img[nan]=-1#np.nan
			ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
			layers.append({'method':'imshow', 'args':(img,),\
				'kws':dict(origin='lower', extent=ext, vmin=0, vmax=0.6, cmap='jet'), \
				'colorbar':dict(label='S p')})

		PlotCatalogue._lbmap(self.lrange, self.brange, figname=figname, parts=self.lbmap_parts, dpi=self.dpi, layers=layers)


	##################################################################################################################
	def _gplane(xrange=[-7,15.5], yrange=[-12,17], figname=None, dpi=600, layers=[], R0=8.15):
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
		ax.text(0.1,0.1,'GC', fontsize=font['SMALL'])
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
			axin = inset_axes(ax, width='3%', height='30%', loc='lower left', \
				bbox_to_anchor=(0.10,0.08,1,1), bbox_transform=ax.transAxes)
			cb = fig.colorbar(im, cax=axin, orientation='vertical', pad=0.05)
			axin.text(-1, 1.03, cbkws['label'], transform=axin.transAxes, fontsize=font['SMALL'])
			cb.ax.tick_params(axis='both', labelsize=font['SMALL'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_gplane(self, n=[1,]):
		#PlotCatalogue._gplane(dpi=self.dpi)
		#return

		cat = Catalogue().open('clump_self0.10_equalusenear.cat')
		suffix = '_fromSelf0.10eun'
		cat = cat[np.argsort(cat.SurfaceDensity)]
		R0=8.15

		x=cat.D*np.cos(cat.b*dtor)*np.cos((cat.l-90)*dtor)
		y=cat.D*np.cos(cat.b*dtor)*np.sin((cat.l-90)*dtor)+R0


		if 0 in n:
			#simple distribution
			layers=[]
			figname = 'fig_plane_xy'+suffix
			layers.append({'method':'plot', 'args':(x, y, '.'), \
				'kws':dict(markersize=0.2)})
			#idx = ~(cat.Dnear>cat.Dfar)
			#layers.append({'method':'plot', 'args':(x[idx], y[idx], '.'), \
			#	'kws':dict(markersize=2.2,zorder=0)})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 1 in n:
			#color=mass, size=size
			layers=[]
			figname = 'fig_plane_ms'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=np.log10(cat.mass), cmap='rainbow', vmin=0.5, vmax=4.5, s=np.abs(cat.physz)/3, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='log(Mass [M$_\odot$])')})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 2 in n:
			#color=Surfacedensity, size=size
			layers=[]
			figname = 'fig_plane_Ss'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=np.log10(cat.SurfaceDensity), cmap='rainbow', vmin=0.25, vmax=1.5, s=np.abs(cat.physz)/2, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='log($\Sigma$ [M$_\odot$ pc$^{-2}$])')})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 3 in n:
			layers=[]
			figname = 'fig_plane_vd'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=cat.avm2, cmap='rainbow', vmin=0, vmax=1.5, s=np.abs(cat.physz)/2, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='logArea (M$_\odot$)')})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 4 in n:
			#color=vertical distance
			layers=[]
			figname = 'fig_plane_Ts'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=cat.Tex, cmap='rainbow', vmin=5, vmax=25, s=np.abs(cat.physz)/2, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='Z (kpc)')})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		if 5 in n:
			#color=vertical distance
			layers=[]
			figname = 'fig_plane_zs'+suffix
			layers.append({'method':'scatter', 'args':(x, y), \
				'kws':dict(c=cat.D*np.sin(cat.b*dtor), cmap='RdYlBu', vmin=-0.5, vmax=0.5, s=cat.physz, \
				marker='.', edgecolors='none', alpha=1.0), \
				'colorbar':dict(label='Z (kpc)')})
			PlotCatalogue._gplane(figname=figname, dpi=self.dpi, layers=layers, R0=R0)

		'''
		plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
		plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True
		fig,ax=plt.subplots(figsize=[5,7])
		plt.subplots_adjust(left=0.15,right=0.9,top=0.95,bottom=0.08)
		
		im = ax.scatter(self.cat.D*np.cos((self.cat.l-90)*np.pi/180), self.cat.D*np.sin((self.cat.l-90)*np.pi/180)+8.5, marker='.', \
			c=np.log10(self.cat.mass), cmap='rainbow', vmin=0.5, vmax=4.5, edgecolors='none', alpha=1, \
			s=np.log10(np.abs(self.cat.sz))*2)

		im = ax.scatter(self.cat.D*np.cos((self.cat.l-90)*np.pi/180), self.cat.D*np.sin((self.cat.l-90)*np.pi/180)+8.5, marker='.', \
			c=np.log10(self.cat.SurfaceDensity), cmap='rainbow', vmin=0.25, vmax=1.5, edgecolors='none', alpha=1, \
			s=np.log10(np.abs(self.cat.sz))*2)
		im = ax.scatter(self.cat.D*np.cos((self.cat.l-90)*np.pi/180), self.cat.D*np.sin((self.cat.l-90)*np.pi/180)+8.5, marker='.', \
			c=self.cat.m1[:,1], cmap='rainbow', edgecolors='none', alpha=1, \
			s=2)#np.log10(np.abs(self.cat.sz))*2)
		'''
		
		#im = ax.plot(np.abs(self.cat.D)*np.cos((self.cat.l-90)*np.pi/180), np.abs(self.cat.D)*np.sin((self.cat.l-90)*np.pi/180)+8.5, '.', markersize=0.2)
		#mask = (self.cat.Rgal<8.5) & (self.cat.Rgal>7.5) & (self.cat.SurfaceDensity<8) & (self.cat.D>8)
		#ax.plot(np.abs(self.cat.D[~mask])*np.cos((self.cat.l[~mask]-90)*np.pi/180), np.abs(self.cat.D[~mask])*np.sin((self.cat.l[~mask]-90)*np.pi/180)+8.5, 'r.', markersize=0.2)


	##################################################################################################################
	def plot_point3(self, figname, dpi=400):
		fig,ax=plt.subplots(ncols=3,sharex=True,sharey=True,figsize=[12,4])
		plt.subplots_adjust(left=0.1,right=0.95,top=0.92,bottom=0.12)

		def tmp(ax, cat, Dist):
			X = cat.angsz/2*Dist*1e3 * cat.SurfaceDensity
			Y = cat.avm2#np.sqrt(3)

			inner = (cat.Rgal<8.15)
			usenear = cat.D==cat.Dnear
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
		cat = Catalogue().open('clump_mas.cat')
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
		ax[1].plot([0,1],[100,100],'--',color='limegreen',label='$\sigma_v$=0.78 R$^{0.43}$')
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

		#plt.show()
		plt.savefig('%s.png' % figname, dpi=self.dpi)
		print('Export to %s.png' % figname)		


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
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_point(self, n=[1,]):
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
			figname = None#'fig_point_Rvd'
			h,xe,ye = np.hist2d()
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
	def _hist(values, components, bins, labels, figname=None, dpi=600, xlog=True, ylog=True, **kws):
		#plot a histogram figure
		fig, ax = plt.subplots(figsize=[7,5])

		if xlog: x = np.sqrt(bins[:-1] * bins[1:])
		else: x = (bins[:-1]+bins[1:])/2
		for c,l in zip(components,labels):
			v=values[c]
			v=v[v>0]
			meanv = 10**np.nanmean(np.log10(v))
			hist, edge = np.histogram(v, bins=bins)
			ax.step(x, hist, where='mid', label='%s (%.2f)' % (l, meanv), color='k' if l=='All' else None)
		ax.legend(fontsize=font['MEDIUM'])

		ax.tick_params(axis='both', labelsize=font['SMALL'])
		if 'xlabel' in kws: ax.set_xlabel(kws['xlabel'], fontsize=font['MEDIUM'])
		if 'ylabel' in kws: ax.set_ylabel(kws['ylabel'], fontsize=font['MEDIUM'])
		if xlog: ax.set_xscale('log')
		if ylog: ax.set_yscale('log')

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname)
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_hist(self, n=[6,]):
		#cat=Catalogue().open('clump_self0.10.cat')
		D = np.load('D.npy')
		l = np.load('l.npy')
		mask = (D>1e-2) & ((l<170) | (l>183))
		#print(mask.sum())

		if 0:
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
			figname = 'fig_hist_Sigma_%s' % suffix
			values = np.load('SD.npy')
			print(values[mask].min(),values[mask].max())
			bins = 10.0**np.arange(-0.8, 3.4, 0.05)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='$\Sigma$ (M$_\odot$ pc$^{-2}$)', ylabel='N')

		if 1 in n:
			figname = 'fig_hist_mass_%s' % suffix
			values = np.load('mass.npy')
			print(values[mask].min(),values[mask].max())
			bins = 10.0**np.arange(-5.5, 5.8, 0.2)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='mass (M$_\odot$)', ylabel='N')

		if 2 in n:
			figname = 'fig_hist_R_%s' % suffix
			values = np.load('physz.npy')/2
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins = 10.0**np.arange(-3.2, 1.7, 0.1)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='R (pc)', ylabel='N')

		if 3 in n:
			figname = 'fig_hist_n_%s' % suffix
			values = np.load('n.npy')
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins = 10.0**np.arange(-1, 6, 0.1)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='n (cm$^{-3}$)', ylabel='N')

		if 4 in n:
			figname = 'fig_hist_vd_%s' % suffix
			values = np.load('avm2.npy')
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins = 10.0**np.arange(-1.9, 0.8, 0.05)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='$\sigma_v$ (km s$^{-1}$)', ylabel='N')

		if 5 in n:
			figname = 'fig_hist_ntvd_%s' % suffix
			values = np.load('avntvd.npy')
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins = 10.0**np.arange(-1.9, 0.8, 0.05)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='$\sigma_{nt}$ (km s$^{-1}$)', ylabel='N')

		if 6 in n:
			figname = 'fig_hist_Tex_%s' % suffix
			values = np.load('Tex.npy')
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins = 10.0**np.arange(0, 2, 0.05)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='$T_{ex}$ (K)', ylabel='N')

		if 7 in n:
			figname = None
			values = np.load('gradient.npy')
			values = np.sqrt((values**2).sum(axis=1))
			print(np.nanmin(values[mask]),np.nanmax(values[mask]))
			bins=10**np.linspace(-3.2,3,40)
			#bins = np.linspace(-80,80,100)
			PlotCatalogue._hist(values, components, bins, labels, figname=figname, \
				xlabel='$\Phi_x$', ylabel='N', xlog=True)


	##################################################################################################################
	def _hist2d(X={}, Y={}, C=None, lines=[], figname=None, dpi=600, fitting=True, **kws):
		#plot a 2d histogram figure

		fig,ax=plt.subplots(figsize=[6,6])
		#plt.subplots_adjust(left=0.15,right=0.9,top=0.95,bottom=0.08)

		if C is None:
			im=ax.hist2d(X['data'], Y['data'], bins=[X['bins'], Y['bins']], cmap='gist_heat_r', zorder=1, alpha=1)
		else:
			im=ax.scatter(X['data'],Y['data'],c=C['data'],cmap='rainbow',s=0.2,zorder=0, norm=LogNorm())

		if fitting:
			#fitting
			logX=np.log10(X['data'])
			if 'error' in X: logXerr=X['error']/X['data']/np.log(10)
			else: logXerr=np.zeros(logX.shape)
			logY=np.log10(Y['data'])
			if 'error' in Y: logYerr=Y['error']/Y['data']/np.log(10)
			else: logYerr=np.zeros(logY.shape)
			mask = np.isfinite(logX) & np.isfinite(logY)# & (logX>-1)
			
			import bces.bces as BCES
			a,b,aerr,berr,covab = BCES.bcesp(logX[mask], logXerr[mask], logY[mask], logYerr[mask], np.zeros(mask.sum()))
			print(a,b,aerr,berr,covab)
			usemethod=2
			lines.append((10**b[usemethod], a[usemethod], 'k-'))
			'''
			p = np.polyfit(logX[mask], logY[mask], 1)
			print(p)
			lines.append((10**p[1], p[0], 'k'))
	
			import nmmn.stats
			def func(x): return x[1]*x[0]+x[2]
			fitm = np.array([a[usemethod],b[usemethod]])
			covm = np.array([ (aerr[usemethod]**2,covab[usemethod]), (covab[usemethod],berr[usemethod]**2) ])
			lcb,ucb,xcb=nmmn.stats.confbandnl(logX[mask], logY[mask], func, fitm, covm, 2, 0.954, np.log10(X['bins']))
			plt.fill_between(10**xcb, 10**lcb, 10**ucb, alpha=0.9, facecolor='orange')
			print(xcb)
			print(lcb)
			print(ucb)
			'''

		#other lines
		def loglinear(x, A, s):
			return [A*v**s for v in x]
		for l in lines:
			label = '%s=%.2f' % (Y['label'].split()[0], l[0])
			if l[1]!=0: label += ' %s' % X['label'].split()[0]
			if l[1]!=0 and l[1]!=1: label += '$^{%.2f}$' % l[1]
			x=[1e-10,1e10]
			plt.plot(x, loglinear(x,l[0],l[1]), l[2], label=label)
		plt.legend(loc='lower right')

		#ax.tick_params(axis='both', labelsize=font['SMALL'])
		ax.set_xscale('log')
		ax.set_yscale('log')
		if 'bins' in X: ax.set_xlim(X['bins'][[0,-1]])
		if 'bins' in Y: ax.set_ylim(Y['bins'][[0,-1]])
		if 'label' in X: ax.set_xlabel(X['label'],fontsize=font['MEDIUM'])
		if 'label' in Y: ax.set_ylabel(Y['label'],fontsize=font['MEDIUM'])
		
		#colorbar
		if isinstance(im,tuple): im=im[3]
		axin = inset_axes(ax, width='40%', height='3%', loc='upper left', \
			bbox_to_anchor=(0.05,0,1,0.97), bbox_transform=ax.transAxes)
		cb = fig.colorbar(im, cax=axin, orientation='horizontal', pad=0.05)
		cb.set_label('N' if C is None else C['label'], fontsize=font['SMALL'])
		cb.ax.tick_params(axis='both', labelsize=font['SMALL'])
		
		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_hist2d(self, n=(1,)):
		D = np.load('D.npy')
		l = np.load('l.npy')
		mask = (D>1e-2) & ((l<170) | (l>183))

		#cat=Catalogue().open('clump_self0.10_equalusenear.cat')
		#the size of half maximum
		R = dict(data=np.load('R.npy')[mask], resolution=52/3600*dtor * D[mask]*1e3, \
			bins=10.0**np.arange(-2.4, 1.7, 0.04), label=r'R [pc]')
		#the size of lowest contour
		#R = dict(data=(np.sqrt(np.load('area.npy')/np.pi)*30/3600/180*np.pi*D*1e3)[mask], resolution=52/3600*dtor * D[mask]*1e3, \
		#	bins=10.0**np.arange(-2.4, 1.7, 0.04), label=r'R [pc]')
		vd = dict(data=np.load('avm2.npy')[mask], resolution=np.repeat(0.168/2.355,mask.sum()), \
			bins=10.0**np.arange(-1.2, 0.8, 0.02), label=r'$\sigma_v$ (km s$^{-1}$)')
		SD = dict(data=np.load('SD.npy')[mask], \
			bins=10.0**np.arange(-0.4, 2.7, 0.025), label=r'$\Sigma$ (M$_\odot$ pc$^{-2}$)')
		mass = dict(data=np.load('mass.npy')[mask], \
			bins=10.0**np.arange(-3.0, 5.8, 0.08), label=r'M$_{LTE}$ (M$_\odot$)')
		Tex = dict(data=np.load('Tex.npy')[mask], \
			bins=10.0**np.arange(-0.1, 2.0, 0.01), label=r'T$_ex$ (K)')
		vdensity = dict(data=np.load('n.npy')[mask], \
			bins=10.0**np.arange(-0.1, 6.0, 0.01), label=r'T$_ex$ (K)')

		SDxR = dict(data=SD['data']*R['data'], \
			bins=10.0**np.arange(-3.0, 4.1, 0.05), label=r'$\Sigma$R (M$_\odot$ pc$^{-1}$)')
		Mach = dict(data=(np.load('avntvd.npy')/np.load('tvd.npy'))[mask], \
			bins=10.0**np.arange(-0.4, 1.2, 0.015), label=r'Mach Number of $\sigma_{nt}$')
		vddR2 = dict(data=vd['data']/R['data']**0.5, \
			bins=10.0**np.arange(-1.5, 1.2, 0.02), label=r'$\sigma_v$/R$^{0.5}$ (km s$^{-1}$ pc$^{-0.5}$)')

		MLTE=mass
		Mvir = dict(data=1162.5*R['data']*vd['data']**2, \
			bins=10.0**np.arange(-1.0, 5.0, 0.08), label=r'M$_{vir}$ (M$_\odot$)')
		MJeans = dict(data=17.3*Tex['data']**1.5*vdensity['data']**(-0.5), \
			bins=10.0**np.arange(-0.0, 3.0, 0.02), label=r'M$_{Jeans}$ (M$_\odot$)')
		alpha = dict(data=Mvir['data']/MLTE['data'], \
			bins=10.0**np.arange(-1.0, 3.0, 0.05), label=r'$\alpha_{vir}$')

		ntvd = dict(data=np.load('ntvd.npy')[:,1][mask], \
			bins=10.0**np.arange(-1,0.8,0.01), label=r'Averaged $\sigma_{nt}$ of spectra')
		avntvd = dict(data=np.load('avntvd.npy')[mask], \
			bins=10.0**np.arange(-1,0.8,0.01), label=r'$\sigma_{nt}$ of averaged spectrum')

		if 0 in n:
			figname='fig_hist2d_R_vd'
			PlotCatalogue._hist2d(R, vd, figname=figname, dpi=self.dpi, lines=[(0.48,0.63,'r--')])
		if 1 in n:
			figname='fig_hist2d_R_mass'
			PlotCatalogue._hist2d(R, mass, figname=figname, dpi=self.dpi, lines=[(36.7,2.2,'r--'),(10,2,'g--'),(100,2,'b--'),(1000,2,'y--')])
		if 2 in n:
			figname='fig_hist2d_mass_vd'
			PlotCatalogue._hist2d(mass, vd, figname=figname, dpi=self.dpi, lines=[(0.19,0.27,'r--')])
		if 3 in n:
			figname='fig_hist2d_SDxR_vd'
			PlotCatalogue._hist2d(SDxR, vd, figname=figname, dpi=self.dpi, lines=[(0.23,0.43,'r--')])
		if 4 in n:
			figname='fig_hist2d_SD_vddR2'
			PlotCatalogue._hist2d(SD, vddR2, figname=figname, dpi=self.dpi, lines=[(0.09,0.5,'r--')])

		if 5 in n:
			figname='fig_hist2d_MLTE_Mvir'
			PlotCatalogue._hist2d(mass, Mvir, figname=figname, dpi=self.dpi, lines=[(3,1,'r--'),(1,1,'g--')])
		if 6 in n:
			figname='fig_hist2d_MLET_MJeans'
			PlotCatalogue._hist2d(mass, MJeans, figname=figname, dpi=self.dpi, lines=[(1,1,'g--')])
		if 7 in n:
			figname='fig_hist2d_MLET_alpha'
			PlotCatalogue._hist2d(mass, alpha, figname=figname, dpi=self.dpi, lines=[(3,0,'g--')])
		if 8 in n:
			figname='fig_hist2d_R_Mach'
			PlotCatalogue._hist2d(R, Mach, figname=figname, dpi=self.dpi, lines=[(1,0,'g--')])
		if 9 in n:
			figname='fig_hist2d_avntvd_ntvd'
			PlotCatalogue._hist2d(ntvd, avntvd, figname=figname, dpi=self.dpi, fitting=True, lines=[(1,1,'g--'),(0,1,'r--')])

	##################################################################################################################

	def _binmean(X, Y, ncomp=None, threshold=1, figname=None, dpi=600, nlimit=5):
		#open fits to data
		X['data'] = fits.open(X['datafile'])[0].data#.ravel()
		Y['data'] = fits.open(Y['datafile'])[0].data#.ravel()
		if 'errorfile' in Y: Y['error'] = fits.open(Y['errorfile'])[0].data#.ravel()
		else: Y['error'] = 1
		if ncomp is not None: ncomp['data'] = fits.open(ncomp['datafile'])[0].data#.ravel()
		print('X range from %f to %f' % (np.nanmin(X['data']), np.nanmax(X['data'])))
		print('Y range from %f to %f' % (np.nanmin(Y['data']), np.nanmax(Y['data'])))

		fig,ax = plt.subplots(figsize=(8,6))
		if 'scale' in X: ax.set_xscale(X['scale'])
		if 'scale' in Y: ax.set_yscale(Y['scale'])

		if ncomp is not None:
			#im=ax.plot(X['data'].ravel(), Y['data'].ravel(),'k,', alpha=1)
			im=ax.hist2d(X['data'].ravel(), Y['data'].ravel(), bins=[X['bins'], Y['bins']], cmap='gray_r', zorder=1, alpha=1)#, norm=LogNorm())

			import scipy.ndimage as ndimage
			#nlimit = 5#int(np.nanmax(ncomp['data']))
			ncomp['data'][ncomp['data']>nlimit]=nlimit
			mycmap=cm.get_cmap('rainbow_r',nlimit)
			for i in range(1,nlimit+1):
				print('Ncomp=%i, %i pixels' % (i,(ncomp['data']==i).sum()))
				h,xe,ye = np.histogram2d(X['data'][ncomp['data']==i], Y['data'][ncomp['data']==i],\
					bins=[X['bins'][::1], Y['bins'][::1]])
				xc = np.sqrt(xe[:-1]*xe[1:])
				yc = np.sqrt(ye[:-1]*ye[1:])
				#ax.contour(xc, yc, h.T, levels=[h.max()/2,h.max()*0.75], colors=[mycmap(i-1),], zorder=10, linewidths=0.8)
				h = ndimage.gaussian_filter(h, sigma=(3,5), order=0)
				ax.contour(xc, yc, h.T, levels=[h.max()*0.5,h.max()*0.7, h.max()*0.9], \
					colors=[mycmap(i-1),], zorder=10-i, linewidths=1.5, alpha=0.7)

			#for colorbar, draw hidden scatter
			im = ax.scatter([1e-5]*nlimit, [1e-5]*nlimit, c=[1]*nlimit, cmap=mycmap, vmin=0.5, vmax=nlimit+0.5)

			def bin_XY(x, y, yerr=1, bins=None, xscale='log', yscale='log'):
				bin_edges = np.linspace(np.nanmin(x), np.nanmax(x), 10) if bins is None else bins
				bin_mean = []
				bin_std = []
				goodY = np.isfinite(y) & np.isfinite(yerr)
				for e0,e1 in zip(bin_edges[:-1],bin_edges[1:]):
					idx = (x>=e0) & (x<e1) & goodY
					print('(%.2e, %.2e) %i pixels' % (e0,e1,idx.sum()))
					if idx.sum()>2:
						values = y[idx] if yscale!='log' else np.log10(y[idx])
						if isinstance(yerr, (int,float)):						
							mean, std = np.mean(values), np.std(values)
						else:
							weights = 1/(yerr[idx]/values)**2#1/yerr[idx]**2
							mean = np.average(values, weights=weights)
							std = np.sqrt(np.average((values-mean)**2, weights=weights))
						if yscale=='log':
							std = [10**mean-10**(mean-std),10**(mean+std)-10**mean]
							mean = 10**mean
						else:
							std = [std, std]
					else:
						mean, std = np.nan, [np.nan,np.nan]
					bin_mean.append(mean)
					bin_std.append(std)
				bin_center = (bin_edges[:-1]+bin_edges[1:])/2 if xscale!='log' else np.sqrt(bin_edges[:-1]*bin_edges[1:])
				bin_mean = np.array(bin_mean)
				bin_std = np.array(bin_std)
				return bin_center, bin_mean, bin_std
					
			bin_center, bin_mean, bin_std = bin_XY(X['data'][ncomp['data']==1], Y['data'][ncomp['data']==1], yerr=1, \
				bins=X['meanbins'], xscale=X['scale'], yscale=Y['scale'])
			ax.errorbar(bin_center, bin_mean, yerr=bin_std.T, linestyle='None', linewidth=1, \
				fmt='o', color=mycmap(0), capsize=3, zorder=12)
			#ax.fill_between(bin_center, bin_mean-bin_std[:,0], bin_mean+bin_std[:,1], color=mycmap(0),edgecolor=None,alpha=0.2)
			#ax.plot(bin_center, bin_mean, color=mycmap(0))
					
			#colorbar
			if isinstance(im,tuple): im=im[3]
			axin = inset_axes(ax, width='25%', height='3%', loc='upper left', \
				bbox_to_anchor=(0.05,0,1,0.97), bbox_transform=ax.transAxes)
			cb = fig.colorbar(im, cax=axin, orientation='horizontal', pad=0.05, ticks=np.arange(1,nlimit+1))
			xticklabels = [str(n) for n in range(1,nlimit+1)]
			xticklabels[-1] = r'$\geq$'+xticklabels[-1]
			cb.ax.set_xticklabels(xticklabels)
			cb.ax.tick_params(axis='both', labelsize=font['SMALL'])
			cb.set_label('N$_{component}$', fontsize=font['SMALL'])
			cb.outline.set_visible(False)

			if 'bins' in X: ax.set_xlim(X['bins'][[0,-1]])
			if 'bins' in Y: ax.set_ylim(Y['bins'][[0,-1]])
		else:
			###ViolinPlot for Ncomp
			ncomp=X
			ncomp['data'][ncomp['data']>nlimit]=nlimit
			data=[]
			posi=[]
			for e0,e1 in zip(ncomp['meanbins'][:-1],ncomp['meanbins'][1:]):
				idx = (ncomp['data']>e0) & (ncomp['data']<=e1)
				if idx.sum()>2:
					data.append(Y['data'][idx])
					posi.append((e0+e1)/2)
			ax.violinplot(data, positions=posi, widths=0.8, showmedians=True)
			xticklabels = ['%i' % n for n in posi]
			xticklabels[-1] = r'$\geq$'+xticklabels[-1]
			ax.set_xticks(posi)
			ax.set_xticklabels(xticklabels)

		if 'label' in X: ax.set_xlabel(X['label'],fontsize = font['MEDIUM'])
		if 'label' in Y: ax.set_ylabel(Y['label'],fontsize = font['MEDIUM'])

		if isinstance(figname, str):
			plt.savefig('%s.png' % figname, dpi=dpi)
			print('Export to %s.png' % figname)			
		else:
			plt.show()

	def plot_binmean(self, n=(0,1,2,)):
		Dlimit = '2kpc'	#''/'1kpc'/'2kpc'/'3kpc'/'5kpc'
		fwhm = '7'	#'7'/'10'
		ns = 2048 if fwhm=='7' else 1024
		#Xs
		logmach = dict(datafile='Bfield/mach%s_fwhm%s.fits' % (Dlimit,fwhm), scale='log', \
			bins=10**np.linspace(-0.6, 1.05, 200), meanbins=10**np.linspace(-0.3,0.7,10), \
			label=r'Mach Number')
		mach = dict(datafile='Bfield/mach%s_fwhm%s.fits' % (Dlimit,fwhm), scale='linear', \
			bins=np.linspace(0, 20, 110), meanbins=np.linspace(0,20,40), \
			label=r'Mach Number')

		logntvd = dict(datafile='Bfield/nt%s_fwhm%s.fits' % (Dlimit,fwhm), scale='log', 
			bins=10.0**np.linspace(-1,0.05,200), meanbins=10.0**np.linspace(-1,-0.1,11), \
			label=r'$\sigma_{nonthermal}$ (km s$^{-1}$)')
		ntvd = dict(datafile='Bfield/nt%s_fwhm%s.fits', scale='linear', \
			bins=np.linspace(0, 1.5, 110), meanbins=np.linspace(0,1.2,12), \
			label=r'$\sigma_{nonthermal}$ (km s$^{-1}$)')

		logN = dict(datafile='Bfield/N%s_fwhm%s.fits' % (Dlimit,fwhm), scale='log', \
			bins=10**np.linspace(20, 22, 200), meanbins=10**np.linspace(20,21.6,10), \
			label=r'N (cm$^{-2}$)')

		ncomp = dict(datafile='ncomp%s.fits' % Dlimit, scale='linear', \
			bins=np.linspace(0, 9, 80), meanbins=np.arange(0.5, 8.6, 1.0), \
			label=r'N$_{component}$')

		#Ys
		logSp = dict(datafile='Bfield/interp_nearest_mapSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns),\
			errorfile='Bfield/interp_nearest_mapsigSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='log', \
			bins=10**np.linspace(-1.3, -0.3, 180), label=r'S p ($\degree$)')
		Sp = dict(datafile='Bfield/interp_nearest_mapSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), \
			errorfile='Bfield/interp_nearest_mapsigSp_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='linear', \
			bins=np.linspace(0, 0.4, 100), label=r'S p ($\degree$)')

		logS = dict(datafile='Bfield/interp_nearest_mapS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), \
			errorfile='Bfield/interp_nearest_mapsigS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='log', \
			bins=10.0**np.linspace(-0.1, 1.8, 180), label=r'S ($\degree$)')
		S = dict(datafile='Bfield/interp_nearest_mapS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), \
			errorfile='Bfield/interp_nearest_mapsigS_fwhm%s_ns%s_AngSt1.fits' % (fwhm, ns), scale='linear', \
			bins=np.linspace(0, 25, 85), label=r'S ($\degree$)')

		if 0 in n:
			figname = 'fig_binmean_ntvd_S'
			PlotCatalogue._binmean(logntvd, logS, ncomp, figname=figname)
			figname = 'fig_binmean_ntvd_Sp'
			PlotCatalogue._binmean(logntvd, logSp, ncomp, figname=figname)
		if 1 in n:
			figname = 'fig_binmean_N_S'
			PlotCatalogue._binmean(logN, logS, ncomp, figname=figname)
			figname = 'fig_binmean_N_Sp'
			PlotCatalogue._binmean(logN, logSp, ncomp, figname=figname)
		if 2 in n:
			figname = 'fig_binmean_mach_S'
			PlotCatalogue._binmean(logmach, logS, ncomp, figname=figname)
			figname = 'fig_binmean_mach_Sp'
			PlotCatalogue._binmean(logmach, logSp, ncomp, figname=figname)
		if 3 in n:
			figname = 'fig_binmean_ncomp_S'
			PlotCatalogue._binmean(ncomp, logS, figname=figname, nlimit=8)
			figname = 'fig_binmean_ncomp_Sp'
			PlotCatalogue._binmean(ncomp, logSp, figname=figname, nlimit=8)


if __name__ == '__main__':

	a=PlotCatalogue()#catalogue='label/035.208+2.042_clump_nt.cat')
	#a.plot_lbmap()
	#a.plot_gplane()
	#a.plot_point3('fig_larson_distance')
	#a.plot_point_Rdv()
	#a.plot_point_mr('fig_MR')
	#a.plot_point()
	#a.plot_hist()
	#a.plot_hist2d()
	a.plot_binmean()
