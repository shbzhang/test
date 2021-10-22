import matplotlib.pyplot as plt
from cubemoment import cubemoment
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

if 0:
	cubemoment('117-120-4-7_U.fits',[-10,-4],zeroth_only=True)
	cubemoment('new117-120-4-7_U.fits',[-10,-4],zeroth_only=True)

if 1:
	old=fits.open('117-120-4-7_U_m0.fits')[0]
	new=fits.open('new117-120-4-7_U_m0.fits')[0]
	res=new.copy()
	res.data = new.data-old.data
	res.writeto('residual.fits',overwrite=True)
	wcs=WCS(new.header,naxis=2)
	ext = wcs.pixel_to_world([-0.5,new.header['NAXIS1']-0.5],[-0.5,new.header['NAXIS2']-0.5])
	ext = [ext[0].l.value,ext[1].l.value,ext[0].b.value,ext[1].b.value]

	fig,ax = plt.subplots(ncols=3,sharex=True,sharey=True,figsize=(8,4.8))
	im = [None]*3
	kws=dict(origin='lower',extent=ext)
	ax[0].set_title('30" sampling')
	im[0] = ax[0].imshow(old.data, vmin=0, vmax=30,**kws)
	ax[1].set_title('30"cos(b) sampling')
	im[1] = ax[1].imshow(new.data, vmin=0, vmax=30,**kws)
	ax[2].set_title('Residual')
	im[2] = ax[2].imshow(res.data, vmin=-.5, vmax=.5,**kws,cmap='RdBu')
	for i,a in zip(im,ax):
		fig.colorbar(i,ax=a,location='bottom')
		a.set_xlabel('Galactic Longitude (deg)')
	ax[0].set_ylabel('Galactic Latitude (deg)')
	#whole
	ax[0].set_xlim([119.5,117])
	ax[0].set_ylim([4,7.25])
	plt.savefig('map0.ps')
	#local1
	ax[0].set_xlim([117.4,117.05])
	ax[0].set_ylim([6.05,6.3])
	plt.savefig('map1.ps')
	#local2
	ax[0].set_xlim([118.8,118.2])
	ax[0].set_ylim([6,6.4])
	plt.savefig('map2.ps')
	#plt.show()

if 0:
	old=fits.open('117-120-4-7_U_rms.fits')[0]
	new=fits.open('new117-120-4-7_U_rms.fits')[0]
	res=new.copy()
	res.data = new.data-old.data

	wcs=WCS(new.header,naxis=2)
	ext = wcs.pixel_to_world([-0.5,new.header['NAXIS1']-0.5],[-0.5,new.header['NAXIS2']-0.5])
	ext = [ext[0].l.value,ext[1].l.value,ext[0].b.value,ext[1].b.value]

	fig,ax = plt.subplots(ncols=3,sharex=True,sharey=True,figsize=(8,4.8))
	im = [None]*3
	kws=dict(origin='lower',extent=ext)
	ax[0].set_title('30" sampling')
	im[0] = ax[0].imshow(old.data, vmin=0.4, vmax=0.6,**kws)
	ax[1].set_title('30"cos(b) sampling')
	im[1] = ax[1].imshow(new.data, vmin=0.4, vmax=0.6,**kws)
	ax[2].set_title('Residual')
	im[2] = ax[2].imshow(res.data, vmin=-0.05, vmax=0.05,**kws,cmap='RdBu')
	for i,a in zip(im,ax):
		fig.colorbar(i,ax=a,location='bottom')
		a.set_xlabel('Galactic Longitude (deg)')
	ax[0].set_ylabel('Galactic Latitude (deg)')
	ax[0].set_xlim([119.5,117])
	ax[0].set_ylim([4,7.25])
	ax[2].plot([119.48,117.02,117.02,119.48,119.48],[5.88,5.88,7.13,7.13,5.88],'r--')
	plt.savefig('rms.ps')
	#plt.show()

if 0:
	from matplotlib.gridspec import GridSpec
	old=fits.open('117-120-4-7_U_rms.fits')[0]
	new=fits.open('new117-120-4-7_U_rms.fits')[0]
	wcs=WCS(new.header,naxis=2)
	ext = wcs.pixel_to_world([0,new.header['NAXIS1']-1],[0,0])
	ext = np.linspace(ext[0].l.value,ext[1].l.value,421)
	fig = plt.figure(constrained_layout=True,figsize=(8,4.8))
	gs = GridSpec(4, 1, figure=fig)
	ax1 = fig.add_subplot(gs[:3,:])
	ax2 = fig.add_subplot(gs[3:,:])
	ax1.plot(ext,np.nanmean(old.data[286:436],axis=0),label='30" sampling')
	ax1.plot(ext,np.nanmean(new.data[286:436],axis=0),label='30"cos(b) sampling')
	ax1.plot(0,0,label='30"cos(b) sampling - 30" sampling')
	ax1.set_ylim(0.397,0.547)
	ax2.plot(0,10)
	ax2.plot(0,10)
	ax2.plot([ext[0],ext[-1]],[0.,0.],'r--')
	ax2.plot(ext,np.nanmean(new.data[286:436],axis=0)-np.nanmean(old.data[286:436],axis=0),label='30" sampling - 30"cos(b) sampling +0.4')
	ax2.set_ylim(-0.027,0.023)
	#broken axis
	ax1.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.xaxis.tick_top()
	ax1.tick_params(labeltop=False)
	ax2.xaxis.tick_bottom()
	d = .012  # how big to make the diagonal lines in axes coordinates
	kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
	ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d, +d), (1 - d*3, 1 + d*3), **kwargs)  # bottom-left diagonal
	ax2.plot((1 - d, 1 + d), (1 - d*3, 1 + d*3), **kwargs)  # bottom-right diagonal
	#other settings
	ax2.set_xlabel('Galactic Longitude (deg)')
	ax1.set_ylabel('RMS (K)')
	ax1.set_xlim([119.75,116.75])
	ax2.set_xlim([119.75,116.75])
	ax1.legend()
	plt.savefig('rms1.ps')
	#plt.show()
