import os,glob,sys,getpass
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from astropy.wcs import WCS
sys.path.append(os.path.abspath('../DeepOutflow/procedure/'))
from regulatetable import Catalogue, Sample
_user = getpass.getuser()

from os.path import exists


def m2correction_simulation():
	#simulate calculate moment2 while intensity below threshold is removed.
	def gauss(x, snr, x0, sigma):
		return snr*np.exp(-(x-x0)**2/2/sigma**2) + np.random.normal(0,1,x.size)
	f = open('m2correction3sigma.dat','w')
	x=np.arange(401)-200
	x0=0.
	threshold=3
	Sigma=np.arange(1,60,0.5)
	SNR=np.arange(3,81,0.5)
	#Fitting=np.zeros((Sigma.size, SNR.size))
	for i,sigma in tqdm(enumerate(Sigma)):
		print(sigma)
		for j,snr in enumerate(SNR):
			fitting=[]
			for niter in range(500):
				T = gauss(x, snr, x0, sigma)
				below = np.argwhere(T < threshold)
				T[:below[below<200].max()+1] = 0
				T[below[below>200].min():] = 0
				fitx0 = (T*x).sum()/T.sum()
				figsigma = np.sqrt((T*(x-fitx0)**2).sum() / T.sum())
				fitting.append(figsigma)
			f.write('%f %f %f\n' % (sigma, snr, np.nanmean(fitting)))
			#Fitting[i,j] = np.nanmean(fitting)
	f.close()

#m2correction_simulation()

def m2correction_interpolate(filename='m2correction2sigma.dat'):
	#interpolate on the simulation points
	from scipy import interpolate
	Sigma, SNR, Fitting = np.loadtxt(filename).T
	#f = interpolate.interp2d(SNR, Fitting, Sigma, kind='linear')
	f = interpolate.LinearNDInterpolator(list(zip(SNR, Fitting)), Sigma/Fitting, fill_value=1)
	'''
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	#ax.scatter3D(SNR, Fitting, Sigma/Fitting, c=Sigma, cmap='rainbow', marker='.')
	x = np.arange(5,81,0.25)
	y = np.arange(1,60,0.25)
	xx, yy = np.meshgrid(x, y)
	zz = f(xx,yy)
	#zz = interpolate.griddata(list(zip(SNR, Fitting)), Sigma, (xx, yy), method='nearest')
	#ax.contour3D(xx,yy,zz,50,cmap='rainbow')
	ax.plot_surface(xx, yy, zz)#, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
	ax.set_xlabel('SNR')
	ax.set_ylabel('Moment2')
	ax.set_zlabel('Sigma');
	plt.show()
	'''
	return f



def orient_fitting(moment):
	pass

def gradient_fitting(moment):
	from scipy.linalg import lstsq
	params=[]
	for mmt in moment:
		mask = mmt[4]>0
		w = mmt[4][mask]
		v = mmt[1][mask]
		y,x = np.argwhere(mask).T
		A = np.c_[x*w, y*w, np.ones(x.shape)*w]
		C,R,_,E = lstsq(A, v*w)    # coefficients
		#Z = C[0]*X + C[1]*Y + C[2]
		#print(C,np.sqrt(R/(v.size * (v*w).var())))
		'''
		import matplotlib.pyplot as plt
		plt.imshow(mmt[1],cmap='rainbow',origin='lower')
		ybox = np.argwhere((mmt[4]>0).any(axis=1))
		ybox = (ybox.min(), ybox.max())
		xbox = np.argwhere((mmt[4]>0).any(axis=0))
		xbox = (xbox.min(), xbox.max())
		plt.xlim(xbox[0]-10, xbox[1]+10)
		plt.ylim(ybox[0]-10, ybox[1]+10)
		plt.plot([np.mean(xbox),np.mean(xbox)+C[0]/10], [np.mean(ybox),np.mean(ybox)+C[1]/10])
		plt.show()
		'''
		w = mmt[0][mask]
		x0 = (w*x).sum()/w.sum()
		y0 = (w*y).sum()/w.sum()
		v0 = (w*v).sum()/w.sum()
		params.append([x0,y0,v0,C[0],C[1]])
		return np.array(params)







class MWISPCube():
	if _user=='sz268601':
		fitspath = '/Users/sz268601/Work/DeepOutflow/procedure/prediction/whole'
		outpath = 'label/'
	else:
		fitspath = '/share/public/shbzhang/deepcube/'
		outpath = 'label/'
	overlap = 36
	boundary = ['120.958', '20.917', '-4.082', '4.082']

	def __init__(self, prefix = '027.042+2.042', redo=True):
		MWISPCube._appendlog('Dealing with %s\n' % prefix)
		self.prefix = prefix
		self.redo = redo
		#input files
		self.usbfile = os.path.join(self.fitspath, prefix+'_U.fits')
		self.lsbfile = os.path.join(self.fitspath, prefix+'_L.fits')
		self.usbrmsfile = os.path.join(self.fitspath, prefix+'_U_rms.fits')
		self.lsbrmsfile = os.path.join(self.fitspath, prefix+'_L_rms.fits')
		self.usbrms = np.nanmean(fits.open(self.usbrmsfile)[0].data) if os.path.exists(self.usbrmsfile) else None
		self.lsbrms = np.nanmean(fits.open(self.lsbrmsfile)[0].data) if os.path.exists(self.lsbrmsfile) else None
		#output files
		self.dcpfile = os.path.join(self.outpath, prefix+'_decompose.fits')
		self.caafile = os.path.join(self.outpath, prefix+'_L_out.fits')
		self.mmtfile = os.path.join(self.outpath, prefix+'_L_mmt.fits')
		self.tvwfile = os.path.join(self.outpath, prefix+'_U_tvw.fits')
		self.mmttable = os.path.join(self.outpath, prefix+'_L_mmt.npy')
		self.tvwtable = os.path.join(self.outpath, prefix+'_U_tvw.npy')
		self.dspfile = os.path.join(self.outpath, prefix+'_NT.fits')

		self.shape = fits.open(self.usbrmsfile)[0].data.shape
		self.xmin = 0 if prefix[:7]==self.boundary[0] else self.overlap//2
		self.xmax = self.shape[-1]-1 if prefix[:7]==self.boundary[1] else self.shape[-1]-1-self.overlap//2
		self.ymin = 0 if prefix[7:]==self.boundary[2] else self.overlap//2
		self.ymax = self.shape[-2]-1 if prefix[7:]==self.boundary[3] else self.shape[-2]-1-self.overlap//2

	def _appendlog(line):
		log = open('decompose.log','a')
		log.write(line)
		log.close()

	##################################################################################################################
	def cube2caa(self):
		print("OBSOLETE, USE FellWalker")
		return
		#OBSOLETE, use fellwalker result now (cupid.py)
		if all([os.path.exists(f) for f in (self.lsbfile, self.lsbrmsfile)]):
			print('rms = %f' % self.lsbrms)
			#decompose 13CO datacube
			hdu = fits.open(self.lsbfile)[0]
			label = MWISPCube._cube2lbl(hdu.data[0], min_intensity=3*self.lsbrms, intensity_step=3*self.lsbrms)	#remember to squeeze data
			fits.PrimaryHDU(data=label, header=hdu.header).writeto(self.dcpfile, overwrite=True)
			#filter labels that are small or weak
			###hdu = fits.open(self.lsbfile)[0]
			###label = fits.open(self.dcpfile)[0].data
			clabel = MWISPCube._lbl2caa(label, hdu.data[0], min_pixel=18, min_area=5, min_channel=5, min_peak=self.lsbrms*3)
			print('%i components found in total' % (clabel.max()+1))
			fits.PrimaryHDU(data=clabel, header=hdu.header).writeto(self.caafile, overwrite=True)

	def _cube2lbl(intensity, min_intensity=0.3*2, intensity_step=0.3*3):
		#label intensity from high to low like clumpfind
		#return mask with different numbers indicating different clusters
		mask = intensity >= min_intensity
		idx = np.argwhere(mask)[np.argsort(intensity[mask])][::-1]
		mask = mask.astype(np.int32)
		mask[:]=-1
		ncan = 0
		shape = mask.shape
		def slicei(i, s):
			return slice(0 if i==0 else i-1, s if i==s-1 else i+2)
		#label intensity cube pixel by pixel from high to low
		for i,j,k in tqdm(idx, desc='label_intensity'):
			nearby = mask[slicei(i,shape[0]), slicei(j,shape[1]), slicei(k,shape[2])]
			nearbycan = np.unique(nearby)
			if nearbycan.size == 1:
				#new peak
				mask[i,j,k] = ncan
				ncan += 1
				continue
			mask[i,j,k] = nearbycan[1]
			if nearbycan.size > 2:
				#near at least 2 candidates
				#mask[i,j,k] = nearbycan[1]
				for c in nearbycan[:1:-1]:
					submask = mask == c
					if intensity[submask].max() - intensity[i,j,k] < intensity_step:
						mask[submask] = nearbycan[1]
		return mask

	def _lbl2caa(label, intensity, min_pixel=18, min_area=5, min_channel=5, min_peak=0.3*5):
		#filter labels and keep those fulfill the criteria
		#return mask with clusters sorted as (0,1,2,3,...)
		ncan = 0
		for c in tqdm(np.unique(label)[1:], desc='label_filter'):
			submask = label == c
			#print(intensity.shape,submask.shape,min_peak)
			if (submask.sum() > min_pixel) & \
				(submask.any(axis=0).sum() > min_area) & \
				(submask.any(axis=(1,2)).sum() > min_channel) & \
				(intensity[submask].max() > min_peak):
				label[submask] = ncan
				ncan+=1
			else:
				label[submask] = -1
		return label


	##################################################################################################################
	def caa2mmt(self):
		#calculate moment for each 13CO component
		doit = exists(self.lsbfile) & exists(self.caafile)
		if not self.redo: doit = doit & (not exists(self.mmtfile))
		if doit:
			print('Calculate moment from CAA')
			hdu = fits.open(self.lsbfile)[0]
			caa = fits.open(self.caafile)[0].data
			mmt = MWISPCube._caa2mmt(caa[0], hdu.data[0], hdu.header, self.lsbrms)
			if len(mmt)>0:
				fits.PrimaryHDU(data=mmt, header=hdu.header).writeto(self.mmtfile, overwrite=True)

	def _caa2mmt(caa, intensity, header, rms):
		#find each component, calculate moment and stack into a cube
		dv = np.abs(header['CDELT3'])
		vaxis = (np.arange(header['NAXIS3'])-header['CRPIX3']+1)*header['CDELT3']+header['CRVAL3']
		vaxis = vaxis[:,np.newaxis,np.newaxis]
		nclump = np.nanmax(caa)
		moment=[]
		if np.isnan(nclump): return moment
		for l in tqdm(range(1,int(nclump)+1), desc='label_moment'):
			mask = caa==l#(label==l) & (intensity>rms*2)
			c = np.argwhere(mask.any(axis=(1,2)))
			cslice = slice(c.min(), c.max()+1)
			subcube = intensity[cslice]*mask[cslice]
			sumi = subcube.sum(axis=0, keepdims=True)
			m0 = sumi * dv/1e3
			m1 = (subcube * vaxis[cslice]).sum(axis=0, keepdims=True) / sumi					#moment 1 need at least 3 channels to be valid
			m2 = np.sqrt((subcube * (vaxis[cslice]-m1)**2).sum(axis=0, keepdims=True) / sumi)	#moment 2 need at least 5 channels to be valid
			peak = subcube.max(axis=0, keepdims=True)
			nchan = mask[cslice].sum(axis=0, keepdims=True)
			moment.append(np.vstack((m0,m1,m2,peak,nchan)))
		return np.array(moment)


	##################################################################################################################
	def mmt2tvw(self):
		#get Tpeak, velcity, and width for 12CO
		doit = exists(self.usbfile) & exists(self.mmtfile)
		if not self.redo: doit = doit & (not exists(self.tvwfile))
		if doit:
			print('Calculate TVW from moment')
			hdu = fits.open(self.usbfile)[0]
			mmt = fits.open(self.mmtfile)[0].data
			tvw = MWISPCube._mmt2tvw(mmt, hdu.data[0], hdu.header, width_factor=10)
			hdu.data = tvw
			fits.PrimaryHDU(data=tvw, header=hdu.header).writeto(self.tvwfile, overwrite=True)

	def _mmt2tvw(moment, intensity, header, width_factor=10):
		#find tpeak in 12CO for each component
		moment[:,1] = (moment[:,1]-header['CRVAL3'])/header['CDELT3']+header['CRPIX3']-1	#in channel
		moment[:,2] = moment[:,2]/header['CDELT3']*width_factor	#in channel
		nlabel = moment.shape[0]
		tvw = np.zeros_like(moment[:,:4])
		vaxis = (np.arange(header['NAXIS3'])-header['CRPIX3']+1)*header['CDELT3']+header['CRVAL3']
		def slicec(c, w, s):
			#center, width, shape
			if np.isnan(w) | (w<2): w=2
			return slice(0 if (c-w)<0 else round(c-w), s if c+w>s-1 else round(c+w)+1)
		for l in tqdm(range(nlabel), desc='label_tpeak'):
			ccen = moment[l,1,...]
			cwid = moment[l,2,...]
			idx = np.argwhere(np.isfinite(ccen))
			for y,x in idx:
				try:
					crange = slicec(ccen[y,x], cwid[y,x], header['NAXIS3'])
					subvaxis = vaxis[crange]
					subspec = intensity[crange,y,x]
					tvw[l,0,y,x] = subspec.sum()
					tvw[l,1,y,x] = (subspec*subvaxis).sum() / tvw[l,0,y,x]
					tvw[l,2,y,x] = np.sqrt(subspec*(subvaxis-tvw[l,1,y,x])**2).sum() / tvw[l,0,y,x]	#moment 2
					tvw[l,3,y,x] = subspec.max()	#Tpeak
				except:
					print('Only %i channels' % moment[l,4,y,x])
					MWISPCube._appendlog('Only %i channels in clump %i at (x=%i, y=%i)\n' % (moment[l,4,y,x],l,x,y))
		return tvw

	##################################################################################################################
	def vd2ntvd():
		if all([os.path.exists(f) for f in (self.mmtfile, self.tvwfile)]):
			Tex = 5.53/np.log(1+5.53/(tw[:,:1]+0.819))

	def dispersion(tw, moment):
		ThermalDispersion = 0.05973641*np.sqrt(Tex)	#np.sqrt(con.k_B/2.33/con.u*u.K).to('km/s')

		OneDDispersion = np.hstack((tw[:,1:],moment[:,1:]))/1e3/np.sqrt(3)

		NonTermalDispersion = np.sqrt(OneDDispersion**2 - ThermalDispersion**2)
		return NonTermalDispersion

	##################################################################################################################
	def mmt2table(self):
		doit = exists(self.mmtfile)
		if not self.redo: doit = doit & (not exists(self.mmttable))
		if doit:
			print('Convert mmt to table')
			mmt = fits.open(self.mmtfile)[0].data
			mask = mmt[:,-1]>0
			table = MWISPCube._sparsearray2table(mmt, mask)
			np.save(self.mmttable, table)

	def tvw2table(self):
		doit = exists(self.mmtfile) & exists(self.tvwfile)
		if not self.redo: doit = doit & (not exists(self.tvwtable))
		if doit:
			print('Convert tvw to table')
			mmt = fits.open(self.mmtfile)[0].data
			mask = mmt[:,-1]>0
			tvw = fits.open(self.tvwfile)[0].data
			table = MWISPCube._sparsearray2table(tvw, mask)
			np.save(self.tvwtable, table)

	def _sparsearray2table(array, mask):
		#convert sparse array to table
		idx = np.argwhere(mask)
		val = [array[:,i][mask] for i in range(array.shape[1])]
		return np.vstack((idx.T,np.array(val)))

	##################################################################################################################
	def mmt2clump(self):
		if os.path.exists(self.mmttable):
			table = np.load(self.mmttable)
			tl,ty,tx,tm0,tm1,tm2,tpk,tnc = table
			clump = []
			for cl in range(tl.max()):
				cmask = tl==cl
				cy = (tm0[cmask]*ty[cmask]).sum()/cm0
				cx = (tm0[cmask]*tx[cmask]).sum()/cm0
				cm0 = tm0[cmask].sum()
				cm1 = tm1[cmask]


	def _c_table2lb(table):
		y=table[1]
		x=table[2]
		ii=table[3]

	def _xy2lb(x, y, header):
		wcs = WCS(header, naxis=2)


if __name__ == '__main__':
	_tmp = MWISPCube()
	if _user=='sz268601':
		usbfiles = ['027.042+2.042']
	else:
		usbfiles = glob.glob(os.path.join(_tmp.fitspath, '[01]*U.fits'))

	prefixes = [os.path.basename(f)[:13] for f in usbfiles]
	for i,prefix in enumerate(prefixes):
		print('>>>[%i/%i]%s<<<' % (i+1,len(prefixes), prefix))
		cube = MWISPCube(prefix=prefix, redo=True)
		#cube.cube2caa()
		#cube.caa2mmt()
		cube.mmt2table()
		#cube.mmt2tvw()
		#cube.tvw2table()
		#cube.mmt2clump()


'''
	#extract clump params to a catalog
	cat = []
	mmttables = glob.glob(os.path.join(outpath,'[01]*mmt.npy'))
	for i,tbl in enumerate(mmttables):
		print('>>>[%i/%i]%s<<<' % (i+1,len(mmttables), tbl))

		np.load
'''

if 0:
	#combine results from all datacubes
	fitsfiles = glob.glob(os.path.join(fitspath, '*_U_rms.fits'))[0:0]
	catfiles = [os.path.join(outpath, os.path.basename(f)[:-11]+'_L_mmt.npy') for f in fitsfiles]
	lons = [float(os.path.basename(f)[:7]) for f in catfiles]
	lats = [float(os.path.basename(f)[7:13]) for f in catfiles]
	if len(fitsfiles)>0:
		xshape=fits.open(fitsfiles[0])[0].header['NAXIS1']
		yshape=fits.open(fitsfiles[0])[0].header['NAXIS2']

	cat=[]
	for i,(ffile,cfile,lon,lat) in enumerate(zip(fitsfiles, catfiles, lons, lats)):
		print('>>>[%i/%i]%s<<<' % (i+1,len(catfiles), ffile))
		if not os.path.exists(cfile): continue
		xmin = 0 if lon==max(lons) else 18
		xmax = xshape-1 if lon==min(lons) else xshape-1-18
		ymin = 0 if lat==min(lats) else 18
		ymax = yshape-1 if lat==max(lats) else yshape-1-18

		x,y,lbl,m0,m1,m2,pk,nc = np.load(cfile).T
		mask = (x>=xmin) & (x<=xmax) & (y>=ymin) & (y<=ymax) & (nc>5)

		cat += list(m2[mask])
		'''
		wcs = WCS(fits.open(ffile)[0].header, naxis=2)
		print(dir(wcs))
		l,b = wcs.pixel_to_world_values(x,y)
		'''
	if len(cat)>0:
		np.save('dv.npy',cat)
		print(len(cat))
		import matplotlib.pyplot as plt
		plt.hist(cat,bins=200)
		plt.plot([168/2.35,168/2.35],[1,1e5],'--')
		plt.yscale('log')
		plt.show()
	'''
	import sys
	sys.path.append('/Users/sz268601/Work/GuideMap/')
	import tile
	#tile.tile(glob.glob('0*_intnum.fits'),'tile_intnum.fits')
	#tile.tile(glob.glob('0*_mmt_m1.fits'),'tile_mmt_m1.fits')
	#tile.tile(glob.glob('0*_mmt_m2.fits'),'tile_mmt_m2.fits')
	#tile.tile(glob.glob('0*_tw_tpeak.fits'),'tile_tw_tpeak.fits')
	#tile.tile(glob.glob('0*_tw_m2.fits'),'tile_tw_m2.fits')
	#tile.tile(glob.glob('0*_12CO.fits'),'tile_NT_12CO.fits')
	#tile.tile(glob.glob('0*_13CO.fits'),'tile_NT_13CO.fits')
	
	print('%i pixels, median=%f' % (len(tnt),np.median(tnt)))
	import matplotlib.pyplot as plt
	plt.hist(tnt, bins=30, range=[0,1])
	plt.plot([np.median(tnt)]*2,[0,200],'--')
	'''
	#plt.show()


#import matplotlib.pyplot as plt

'''
x=np.arange(61)
for snr in np.arange(5,81,0.5):
	plt.plot(Sigma[SNR==snr],Fitting[SNR==snr])
	a, b, c = np.polyfit(Sigma[SNR==snr],Fitting[SNR==snr],deg=2)
	print(a,b,c)
	plt.plot(x,a*x**2+b*x+c)
'''
'''
ax = plt.scatter(SNR, Fitting/Sigma, c=Sigma, cmap='rainbow', marker='+')
SNR=np.arange(5,81)
g = [gausscut(2/s)**1.3 for s in SNR]
plt.plot(SNR,g, 'k')
plt.colorbar(ax)
'''
#plt.imshow(Fitting, cmap='rainbow')
#print(Fitting.shape,Sigma.shape)
#print(SNR)
#for f in Fitting.T:
#	plt.plot(f,Sigma,'.')
#plt.scatter(Sigma, Sigma/Fitting, c=SNR)
#plt.plot([0,Sigma.max()],[0,Sigma.max()],'--')
#plt.show()


