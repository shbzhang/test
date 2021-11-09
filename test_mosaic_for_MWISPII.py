#this code compare the difference of mosaicked datacube between using 30"cos(b0) and using 30" spacing

import matplotlib.pyplot as plt
from cubemoment import cubemoment
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

do_differentVrange=0
do_histofresidual=0
do_spectrallines=0
do_IIresidual = 0
do_RMSresidual = 0
do_RMSmeanalongGB = 0


if do_differentVrange:
	#testing different integrating range
	cubemoment('new117-120-4-7_U.fits',[-100,100],zeroth_only=True)
	wide = fits.open('new117-120-4-7_U_m0.fits')[0]
	cubemoment('new117-120-4-7_U.fits',[-10,10],zeroth_only=True)
	middle = fits.open('new117-120-4-7_U_m0.fits')[0]
	cubemoment('new117-120-4-7_U.fits',[-1,1],zeroth_only=True)
	narrow = fits.open('new117-120-4-7_U_m0.fits')[0]
	wcs=WCS(narrow.header,naxis=2)
	ext = wcs.pixel_to_world([-0.5,narrow.header['NAXIS1']-0.5],[-0.5,narrow.header['NAXIS2']-0.5])
	ext = [ext[0].l.value,ext[1].l.value,ext[0].b.value,ext[1].b.value]

	fig,ax = plt.subplots(ncols=3,sharex=True,sharey=True,figsize=(8,4.8))
	im = [None]*3
	kws=dict(origin='lower',extent=ext)
	ax[0].set_title('[-100, 100] km/s')
	im[0] = ax[0].imshow(wide.data, vmin=-1, vmax=50,**kws)
	ax[1].set_title('[-10, 10] km/s')
	im[1] = ax[1].imshow(middle.data, vmin=-1, vmax=20,**kws)
	ax[2].set_title('[-1, 1] km/s')
	im[2] = ax[2].imshow(narrow.data, vmin=-1, vmax=5,**kws)
	for i,a in zip(im,ax):
		fig.colorbar(i,ax=a,location='bottom')
		a.set_xlabel('Galactic Longitude (deg)')
	ax[0].set_ylabel('Galactic Latitude (deg)')
	#whole
	ax[0].set_xlim([119.5,117])
	ax[0].set_ylim([4,7.25])
	plt.savefig('diffrange.ps')
	#plt.show()

if do_histofresidual:
	#how large is the residual
	def oldnewres(cen, w, ax):
		rms = 0.5*np.sqrt(0.158737644553*w)
		cubemoment('117-120-4-7_U.fits', [cen-w/2,cen+w/2], zeroth_only=True)
		old = fits.open('117-120-4-7_U_m0.fits')[0]
		cubemoment('new117-120-4-7_U.fits', [cen-w/2,cen+w/2], zeroth_only=True)
		new = fits.open('new117-120-4-7_U_m0.fits')[0]
		res = new.data-old.data
		old = old.data[226:436,31:-30].ravel()#/rms
		new = new.data[226:436,31:-30].ravel()#/rms
		res = res[226:436,31:-30].ravel()/rms
		ax[0].hist(old,bins=100,range=[-20,80])
		ax[0].set_xlim(-20,80)
		ax[1].hist(new,bins=100,range=[-20,80])
		ax[1].set_xlim(-20,80)
		ax[2].hist(res,bins=200,range=[-8,8])
		ax[2].set_xlim(-7,7)
		ax[0].text(0.98,0.98,'%i$\pm$%i km/s' % (cen, w/2),\
			transform=ax[0].transAxes, verticalalignment='top', horizontalalignment='right')
		ax[1].text(0.98,0.98,'%i$\pm$%i km/s' % (cen, w/2),\
			transform=ax[1].transAxes, verticalalignment='top', horizontalalignment='right')
		ax[2].text(0.98,0.98,'mean=%+.2e K km/s\nstd=%.2f RMS\nRMS=%.2f K km/s\n%i$\pm$%i km/s' % (rms*res.mean(), np.std(res), rms, cen, w/2),\
			transform=ax[2].transAxes, verticalalignment='top', horizontalalignment='right')
		for a in ax:
			a.set_yscale('log')

	fig,ax = plt.subplots(ncols=3,nrows=3,sharex=False,sharey=True,figsize=(8,5))
	#kws=dict(origin='lower',extent=ext)
	oldnewres(-7,4, ax[0])
	oldnewres(-7,20, ax[1])
	oldnewres(-7,100,ax[2])
	ax[0,0].set_title('30" sampling')
	ax[0,1].set_title('30"cos(b) sampling')
	ax[0,2].set_title('Residual')
	for a in ax[:,0]:
		a.set_ylabel('Count')
	ax[2,0].set_xlabel('Integrated intensity (K km/s)')
	ax[2,1].set_xlabel('Integrated intensity (K km/s)')
	ax[2,2].set_xlabel('Residual / RMS')
	ax[0,0].set_ylim(0.6,8e4)
	plt.savefig('hist.ps')
	#plt.show()

if do_spectrallines:
	#compare spectral lines
	old = fits.open('117-120-4-7_U.fits')[0]
	new = fits.open('new117-120-4-7_U.fits')[0]
	wcs = WCS(new.header, naxis=2)
	velocity = (np.arange(new.header['NAXIS3'])-new.header['CRPIX3']+1)*new.header['CDELT3']+new.header['CRVAL3']
	velocity /= 1e3
	fig,ax = plt.subplots(ncols=3,nrows=3,figsize=(8,8))
	#ax=ax.ravel()

	l = [118.5+offset for offset in [0.0, 0.125, 0.25]]
	b = [6.18]*3
	x,y = wcs.world_to_pixel(SkyCoord(l, b, unit='deg', frame='galactic'))
	x,y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
	for i in range(3):
		oldspec = old.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		newspec = new.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		ax[0,i].step(velocity, oldspec, where='mid', label='30" sampling', linewidth=2)
		ax[0,i].step(velocity, newspec, where='mid', label='30"cos(b) sampling', linewidth=1.2)
		ax[0,i].text(0.02, 0.02, 'l=%.3f b=%.2f' % (l[i],b[i]), transform=ax[0,i].transAxes)
		ax[0,i].set_xlim([-15, 0])
		#ax[0,i].set_ylim([-1, 12.5])
	l = [117.5+offset for offset in [0.0, 0.125, 0.25]]
	b = [6.31]*3
	x,y = wcs.world_to_pixel(SkyCoord(l, b, unit='deg', frame='galactic'))
	x,y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
	for i in range(3):
		oldspec = old.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		newspec = new.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		ax[1,i].step(velocity, oldspec, where='mid', label='30" sampling', linewidth=2)
		ax[1,i].step(velocity, newspec, where='mid', label='30"cos(b) sampling', linewidth=1.2)
		ax[1,i].text(0.02, 0.02, 'l=%.3f b=%.2f' % (l[i],b[i]), transform=ax[1,i].transAxes)
		ax[1,i].set_xlim([-8, 7])
		#ax[1,i].set_ylim([-1, 6])
	l = [117+offset for offset in [0.0, 0.125, 0.25]]
	#b = [6.75]*3
	b = [5.25]*3
	x,y = wcs.world_to_pixel(SkyCoord(l, b, unit='deg', frame='galactic'))
	x,y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
	for i in range(3):
		oldspec = old.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		newspec = new.data[:, y[i]-1:y[i]+2, x[i]-1:x[i]+2].mean(axis=-1).mean(axis=-1)
		ax[2,i].step(velocity, oldspec, where='mid', label='30" sampling', linewidth=2)
		ax[2,i].step(velocity, newspec, where='mid', label='30"cos(b) sampling', linewidth=1.2)
		ax[2,i].text(0.02, 0.02, 'l=%.3f b=%.2f' % (l[i],b[i]), transform=ax[2,i].transAxes)
		ax[2,i].set_xlim([-105, 75])
		ax[2,i].set_ylim([-0.9, 0.9])

	ax[0,0].set_title('Cell center')
	ax[0,1].set_title('In the middle')
	ax[0,2].set_title('Cell edge')
	[ax[i,0].set_ylabel('Tmb (K)') for i in range(3)]
	[ax[-1,i].set_xlabel('velocity (km/s)') for i in range(3)]
	ax[2,2].legend(loc=1,prop={'size':8})
	#whole
	plt.savefig('spec.ps')
	#plt.show()

if do_IIresidual:
	ran,vmax = [-10,-4],30
	#ran,vmax = [-100,100],3
	#ran,vmax = [-25,4],70
	cubemoment('117-120-4-7_U.fits',ran,zeroth_only=True)
	cubemoment('new117-120-4-7_U.fits',ran,zeroth_only=True)

	#compare integrated intensity map
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
	im[0] = ax[0].imshow(old.data, vmin=-2, vmax=vmax,**kws)
	ax[1].set_title('30"cos(b) sampling')
	im[1] = ax[1].imshow(new.data, vmin=-2, vmax=vmax,**kws)
	ax[2].set_title('Residual')
	im[2] = ax[2].imshow(res.data, vmin=-.5, vmax=.5,**kws,cmap='RdBu')
	ax[2].text(0.98,0.02, '[%i, %i] km/s' % tuple(ran), transform=ax[2].transAxes, verticalalignment='bottom', horizontalalignment='right')
	for i,a in zip(im,ax):
		fig.colorbar(i,ax=a,location='bottom')
		a.set_xlabel('Galactic Longitude (deg)')
	ax[0].set_ylabel('Galactic Latitude (deg)')
	#whole
	ax[0].set_xlim([119.5,117])
	ax[0].set_ylim([4,7.25])
	plt.savefig('map0.ps')
	#local1
	#ax[0].set_xlim([118.9,117.8])
	#ax[0].set_ylim([4.9,5.5])
	#plt.savefig('map1.ps')
	#local2
	#ax[0].set_xlim([118.8,118.2])
	#ax[0].set_ylim([6,6.4])
	#plt.savefig('map2.ps')
	plt.show()

if do_RMSresidual:
	#compare rms
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
	ax[2].plot([119.48,117.02,117.02,119.48,119.48],[5.38,5.38,7.13,7.13,5.38],'r--')
	plt.savefig('rms.ps')
	#plt.show()

if do_RMSmeanalongGB:
	#compare rms averaged along GB
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
	ax1.plot(ext,np.nanmean(old.data[226:436],axis=0),label='30" sampling')
	ax1.plot(ext,np.nanmean(new.data[226:436],axis=0),label='30"cos(b) sampling')
	ax1.plot(0,0,label='30"cos(b) sampling - 30" sampling')
	ax1.set_ylim(0.397,0.547)
	ax2.plot(0,10)
	ax2.plot(0,10)
	ax2.plot([ext[0],ext[-1]],[0.,0.],'r--')
	ax2.plot(ext,np.nanmean(new.data[226:436],axis=0)-np.nanmean(old.data[226:436],axis=0),label='30" sampling - 30"cos(b) sampling +0.4')
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

