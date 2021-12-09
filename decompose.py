import os,glob,sys,getpass
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from astropy.wcs import WCS
sys.path.append(os.path.abspath('../DeepOutflow/procedure/'))
from regulatetable import Catalogue, Sample
_user = getpass.getuser()

from os.path import exists
from scipy.linalg import lstsq

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors


def m2correction_simulation():
	#simulate calculate moment2 while intensity below threshold is removed.
	def gauss(snr, x0, sigma):
		hw = int(sigma*2.355/2*3)
		x = np.arange(hw*2+1)-hw
		return x, snr*np.exp(-(x-x0)**2/2/sigma**2) + np.random.normal(0,1,x.size), hw
	f = open('m2correction2sigma.dat','w')
	x=np.arange(601)-300
	x0=0.
	threshold=2
	#Sigma=np.arange(1,60,0.5)
	#SNR=np.arange(2,81,0.5)
	SNR=2**np.linspace(1.6,6.5,20)
	for snr in tqdm(SNR):
		#upper = np.exp(np.exp(1.25-np.log(np.log(snr))*3))
		Sigma = 2**np.linspace(0,6,20)
		for sigma in Sigma:
			fitting=[]
			for niter in range(50):
				x,T,hw = gauss(snr, x0, sigma)
				below = T < threshold
				#maxjump=4
				below = below & np.roll(below,1) & np.roll(below,2) & np.roll(below,3)
				below = below | np.roll(below,-1) | np.roll(below,-2) | np.roll(below,-3)
				#cleaniter=1
				below = below | np.roll(below,-1) | np.roll(below,1)
				below = np.argwhere(below)
				try:
					T[:below[below<hw].max()+1] = 0
					T[below[below>hw].min():] = 0
				except:
					niter-=1
					#print(below)
				fitx0 = (T*x).sum()/T.sum()
				fitsigma = np.sqrt((T*(x-fitx0)**2).sum() / T.sum())
				fitting.append(fitsigma)
			f.write('%f %f %f\n' % (sigma, snr, np.nanmean(fitting)))
			#Fitting[i,j] = np.nanmean(fitting)
	f.close()

'''
###Plot an example
def gauss(snr, x0, sigma):
	#np.random.seed(42)
	hw = int(sigma*2.355/2*5)
	x = np.arange(hw*2+1)-hw
	return x, snr*np.exp(-(x-x0)**2/2/sigma**2) + np.random.normal(0,1,x.size), hw
x,T,hw = gauss(10, 0.1, 30)
plt.step(x,T)
below = T < 2
#maxjump=4
below = below & np.roll(below,1) & np.roll(below,2) & np.roll(below,3)
below = below | np.roll(below,-1) | np.roll(below,-2) | np.roll(below,-3)
#cleaniter=1
below = below | np.roll(below,-1) | np.roll(below,1)
below = np.argwhere(below)
T[:below[below<hw].max()+1] = 0
T[below[below>hw].min():] = 0
plt.bar(x,T,1,alpha=0.5,color='orange')
plt.plot(x,x*0,'r--')
fitx0 = (T*x).sum()/T.sum()
fitsigma = np.sqrt((T*(x-fitx0)**2).sum() / T.sum())
plt.xlabel('Channel')
plt.ylabel('SNR')
plt.show()
'''

def m2correction_interpolate(filename='m2correction2sigma.dat', plot=False):
	#interpolate on the simulation points
	from scipy import interpolate
	Sigma, SNR, Fitting = np.loadtxt(filename).T
	#f = interpolate.interp2d(SNR, Fitting, Sigma, kind='linear')
	f = interpolate.LinearNDInterpolator(list(zip(SNR, Fitting)), Sigma/Fitting)#, fill_value=1)
	if plot:
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.scatter3D(SNR, Fitting, Sigma/Fitting, c=Sigma, cmap='rainbow', marker='.')
		x = np.arange(2,81,0.25)
		y = np.arange(1,60,0.25)
		xx, yy = np.meshgrid(x, y)
		zz = f(xx,yy)
		#zz = interpolate.griddata(list(zip(SNR, Fitting)), Sigma, (xx, yy), method='nearest')
		#ax.contour3D(xx,yy,zz,levels=np.arange(1,20,0.5),cmap='rainbow')
		#ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
		ax.set_xlabel('SNR')
		ax.set_ylabel('Moment2')
		ax.set_zlabel('Sigma/Fitting');
		plt.show()
	return f



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
		#output files
		self.dcpfile = os.path.join(self.outpath, prefix+'_decompose.fits')
		self.caafile = os.path.join(self.outpath, prefix+'_L_out.fits')
		self.mmtfile = os.path.join(self.outpath, prefix+'_L_mmt.fits')
		self.tvwfile = os.path.join(self.outpath, prefix+'_U_tvw.fits')
		self.mmttable = os.path.join(self.outpath, prefix+'_L_mmt.npy')
		self.mmtctable = os.path.join(self.outpath, prefix+'_L_mmtCM2.npy')
		self.tvwtable = os.path.join(self.outpath, prefix+'_U_tvw.npy')
		self.dspfile = os.path.join(self.outpath, prefix+'_NT.fits')

		self.clumptable = os.path.join(self.outpath, prefix+'_clump.cat')
		self.cdsttable = os.path.join(self.outpath, prefix+'_clump_dst.cat')
		self.cmastable = os.path.join(self.outpath, prefix+'_clump_mas.cat')
		self.cgrdtable = os.path.join(self.outpath, prefix+'_clump_grd.cat')

		self.m2map = os.path.join(self.outpath, prefix+'_map_m2.fits')

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
		if exists(self.lsbfile) & exists(lsbrmsfile):
			print('rms = %f' % self.lsbrms)
			#decompose 13CO datacube
			hdu = fits.open(self.lsbfile)[0]
			self.lsbrms = np.nanmean(fits.open(self.lsbrmsfile)[0].data) if exists(self.lsbrmsfile) else None
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
			self.lsbrms = np.nanmean(fits.open(self.lsbrmsfile)[0].data) if exists(self.lsbrmsfile) else None
			mmt = MWISPCube._caa2mmt(caa[0], hdu.data[0], hdu.header, self.lsbrms)
			if len(mmt)>0:
				fits.PrimaryHDU(data=mmt, header=hdu.header).writeto(self.mmtfile, overwrite=True)

	def _caa2mmt(caa, intensity, header, rms):
		#find each component, calculate moment and stack into a cube
		dv = np.abs(header['CDELT3'])/1e3
		vaxis = (np.arange(header['NAXIS3'])-header['CRPIX3']+1)*header['CDELT3']+header['CRVAL3']
		vaxis = vaxis[:,np.newaxis,np.newaxis]/1e3	#to km/s
		nclump = np.nanmax(caa)
		moment=[]
		if np.isnan(nclump): return moment
		for l in tqdm(range(1,int(nclump)+1), desc='label_moment'):
			mask = caa==l#(label==l) & (intensity>rms*2)
			c = np.argwhere(mask.any(axis=(1,2)))
			cslice = slice(c.min(), c.max()+1)
			subcube = intensity[cslice]*mask[cslice]
			sumi = subcube.sum(axis=0, keepdims=True)
			m0 = sumi * dv																		#in K*km/s
			m1 = (subcube * vaxis[cslice]).sum(axis=0, keepdims=True) / sumi					#in km/s, moment 1 need at least 3 channels to be valid
			m2 = np.sqrt((subcube * (vaxis[cslice]-m1)**2).sum(axis=0, keepdims=True) / sumi)	#in km/s, moment 2 need at least 5 channels to be valid
			peak = subcube.max(axis=0, keepdims=True)	#in K
			nchan = mask[cslice].sum(axis=0, keepdims=True)	#in channel
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
		moment[:,1] = (moment[:,1]*1e3-header['CRVAL3'])/header['CDELT3']+header['CRPIX3']-1	#in channel
		moment[:,2] = moment[:,2]*1e3/header['CDELT3']*width_factor	#in channel
		nlabel = moment.shape[0]
		tvw = np.zeros_like(moment[:,:4])
		dv = np.abs(header['CDELT3'])/1e3
		vaxis = (np.arange(header['NAXIS3'])-header['CRPIX3']+1)*header['CDELT3']+header['CRVAL3']
		vaxis /= 1e3
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
					sumi = subspec.sum()
					tvw[l,0,y,x] = sumi * dv 												#in K*km/s
					tvw[l,1,y,x] = (subspec*subvaxis).sum() / sumi							#in km/s
					tvw[l,2,y,x] = np.sqrt(subspec*(subvaxis-tvw[l,1,y,x])**2).sum() / sumi	#in km/s, moment 2
					tvw[l,3,y,x] = subspec.max()	#in K, Tpeak
				except:
					print('Only %i channels' % moment[l,4,y,x])
					MWISPCube._appendlog('Only %i channels in clump %i at (x=%i, y=%i)\n' % (moment[l,4,y,x],l,x,y))
		return tvw


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
	def table_correctm2(self, corr=lambda x,y: 1):
		doit = exists(self.mmttable) & exists(self.lsbrmsfile)
		if not self.redo: doit = doit & (not exists(self.mmtctable))
		if doit:
			print('Correct M2')
			mmt = np.load(self.mmttable)
			hdu = fits.open(self.lsbrmsfile)[0]
			self.lsbrms = np.nanmean(hdu.data)
			snr = mmt[6]/self.lsbrms
			fit = mmt[5]*1e3/hdu.header['CDELT3']
			factor = corr(snr,fit)
			mmt[5] *= factor
			np.save(self.mmtctable, mmt)
			#print(snr[np.isnan(factor)], fit[np.isnan(factor)])
			#return snr[np.isnan(factor)], fit[np.isnan(factor)]


	##################################################################################################################
	def mmttvw2clump(self, filename=None):
		doit = exists(self.mmtctable) & exists(self.tvwtable) & exists(self.usbrmsfile)
		if not self.redo: doit = doit & (not exists(self.clumptable))
		if doit:
			print('Convert mmt table to clump table')
			wcs = WCS(fits.open(self.usbrmsfile)[0].header, naxis=2)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmtctable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			cat = []
			for cn in range(int(mmtn.max())+1):
				cmask = mmtn==cn
				clm0 = mmtm0[cmask].sum()	#in
				cum0 = tvwm0[cmask].sum()
				cy = (mmtm0[cmask]*mmty[cmask]).sum()/clm0	#in array indices
				if (cy<self.ymin) | (cy>self.ymax): continue
				cx = (mmtm0[cmask]*mmtx[cmask]).sum()/clm0	#in array indices
				if (cx<self.xmin) | (cx>self.xmax): continue
				cysz = np.sqrt((mmtm0[cmask]*(mmty[cmask]-cy)**2).sum()/clm0) #in pixel
				cxsz = np.sqrt((mmtm0[cmask]*(mmtx[cmask]-cx)**2).sum()/clm0) #in pixel
				nc = mmtnc[cmask]>3
				clm1 = (mmtm0[cmask][nc] * mmtm1[cmask][nc]).sum() / mmtm0[cmask][nc].sum()	#in km/s
				cum1 = (tvwm0[cmask][nc] * tvwm1[cmask][nc]).sum() / tvwm0[cmask][nc].sum()	#in km/s
				nc = mmtnc[cmask]>5
				clm2 = (mmtm0[cmask][nc] * mmtm2[cmask][nc]).sum() / mmtm0[cmask][nc].sum()	#in km/s
				cum2 = (tvwm0[cmask][nc] * tvwm2[cmask][nc]).sum() / tvwm0[cmask][nc].sum() #in km/s
				clpk = mmtpk[cmask].max()	#in K
				cupk = tvwpk[cmask].max()	#in K
				cnc = mmtnc[cmask].sum()	#in voxel
				cl,cb = wcs.pixel_to_world_values(cx,cy)	#in deg
				cat.append({'prefix':self.prefix, 'num':cn, 'x':cx, 'y':cy, 'l':float(cl), 'b':float(cb), \
					'sx':cxsz, 'sy':cysz, \
					'nc':cnc, 'm0':[cum0, clm0], 'm1':[cum1, clm1], 'm2':[cum2, clm2], 'peak':[cupk, clpk]})
			cat = Catalogue(cat)
			cat.write(self.clumptable)


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	def orient_fitting(self):
		pass


	##################################################################################################################
	def clump_distance(self):
		doit = exists(self.clumptable)
		if not self.redo: doit = doit & (not exists(self.cdsttable))
		if doit:
			print('Add distance info to clump table')
			cat = Catalogue().open(self.clumptable)
			if len(cat)==0:return

			#P. Mroz et al 2018
			dtor=np.pi/180
			R0 = 8.5#kpc
			Vr = 233.6#-1.34*r
			V0 = 233.6-1.34*R0
			Vp = cat.m1[:,1] / np.cos(cat.b*dtor)
			l=cat.l*dtor
			beam = 50#arcsec

			#Marc-Antoine et al. 2017
			factor = R0 * np.sin(l) / (Vp+V0*np.sin(l))
			Rgal = 233.6*factor / (1+1.34*factor)

			Dnear = R0 * np.cos(l) - np.sqrt(Rgal**2 - R0**2 * np.sin(l)**2)
			D = R0 * np.cos(l) + np.sqrt(Rgal**2 - R0**2 * np.sin(l)**2)
			#decide between near and far distance
			ang_sz = np.sqrt(np.sqrt((cat.sx*2.355*30)**2 - beam**2) * np.sqrt((cat.sy*2.355*30)**2 - beam**2))/3600*dtor
			dv = cat.m2[:,1]*np.sqrt(3)	#3d vdisp in km/s
			log_sz_larson = np.log10((dv/0.48)**(1/0.63))	#in pc
			log_sz_far = np.log10(ang_sz*D*1e3)
			log_sz_near = np.log10(ang_sz*Dnear*1e3)
			usenear = (Rgal<R0) & (np.abs(log_sz_far - log_sz_larson) > np.abs(log_sz_near - log_sz_larson))
			D[usenear] = Dnear[usenear]
			SZ = ang_sz*D*1e3

			for i,(r,d,sz) in enumerate(zip(Rgal, D, SZ)):
				cat[i].Rgal = r 	#in kpc
				cat[i].D = d 		#in kpc
				cat[i].sz = sz 		#in pc
			cat.write(self.cdsttable)


	##################################################################################################################
	def clump_mass(self):
		doit = exists(self.mmttable) & exists(self.tvwtable) & exists(self.cdsttable)
		if not self.redo: doit = doit & (not exists(self.cgrdtable))
		if doit:
			print('Add mass info to clump table')
			cat = Catalogue().open(self.cdsttable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			#Nagahama et al. 1998
			Tex = 5.53/np.log(1+5.53/(tvwpk+0.819))	#in K
			N = 1.49e20 / (1-np.exp(-5.29/Tex)) * mmtm0	#in cm-2
			for cn in range(int(mmtn.max())+1):
				if cn in cat.num:
					cmask = (mmtn==cn)
					idx = int(np.argwhere(cat.num==cn))
					###N=1*u.Unit('cm-2')
					###D=1*u.Unit('kpc')
					#Tex
					cat[idx].Tex = np.nanmax(Tex[cmask])	#in K
					#column density
					cat[idx].N = np.nanmean(N[cmask]) 	#in cm-2
					####surface density
					###f=(N*con.u*1.36*2).to('solMass/pc2') = 2.16278417e-20
					cat[idx].SurfaceDensity = cat[idx].N*2.16278417e-20		#in Msun/pc2
					####convert factor from N(cm-2) to mass(M_sun) at D(kpc)
					###f=((30/3600*np.pi/180*D)**2 * N * con.u*1.36*2 / con.M_sun).cgs = 4.57515092e-22
					cat[idx].mass = np.nansum(N[cmask])*4.57515092e-22*cat[idx].D**2	#in solar mass
			cat.write(self.cmastable)

	##################################################################################################################
	def clump_gradient(self):
		doit = exists(self.mmttable) & exists(self.cmastable)
		if not self.redo: doit = doit & (not exists(self.cgrdtable))
		if doit:
			print('Add gradient info to clump table')
			cat = Catalogue().open(self.cmastable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			for cn in range(int(mmtn.max())+1):
				if cn in cat.num:
					cmask = (mmtn==cn) & (mmtnc>3)
					x = mmtx[cmask]
					y = mmty[cmask]
					v = mmtm1[cmask]
					w = mmtm0[cmask]
					idx = int(np.argwhere(cat.num==cn))
					C = MWISPCube._gradient_fitting(x, y, v, w)
					C = [c/(30/3600*np.pi/180*cat.D[idx]*1e3) for c in C]
					#print(C)
					cat[idx].gradient = C #in km/s/pc
			cat.write(self.cgrdtable)

	def _gradient_fitting(x, y, v, w):
		A = np.c_[x*w, y*w, np.ones(x.shape)*w]
		C,R,_,E = lstsq(A, v*w)    # coefficients
		return list(C[0:2])

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	def table2image(self):
		doit = exists(self.mmttable) & exists(self.usbrmsfile) & exists(self.cdsttable)
		if not self.redo: doit = doit & (not exists(self.m2map))
		if doit:
			cat = Catalogue().open(self.cdsttable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			hdu = fits.open(self.usbrmsfile)[0]
			d = np.array([cat.D[int(np.argwhere(cat.num==n))] if n in cat.num else np.inf for n in mmtn])
			image = np.zeros_like(hdu.data)
			for x in range(self.shape[-1]):
				for y in range(self.shape[-2]):
					mask = (mmtx==x) & (mmty==y) & (mmtnc>5) & np.isfinite(d)
					if mask.any():
						idx = np.argmin(d[mask])
						image[y,x] = mmtm2[mask][idx]
			fits.PrimaryHDU(data=image, header=hdu.header).writeto(self.m2map, overwrite=True)






if __name__ == '__main__':
	_tmp = MWISPCube()
	if _user=='sz268601':
		#usbfiles = ['027.042+2.042']
		usbfiles = glob.glob(os.path.join(_tmp.fitspath, '[012]*U_rms.fits'))
	else:
		usbfiles = glob.glob(os.path.join(_tmp.fitspath, '[012]*U.fits'))

	#m2correction_simulation()
	#corr = m2correction_interpolate('m2correction2sigma.dat', plot=True)
	prefixes = [os.path.basename(f)[:13] for f in usbfiles][:0]
	for i,prefix in enumerate(prefixes):
		print('>>>[%i/%i]%s<<<' % (i+1,len(prefixes), prefix))
		cube = MWISPCube(prefix=prefix, redo=True)
		###cube.cube2caa()
		#cube.caa2mmt()
		#cube.mmt2table()
		#cube.mmt2tvw()
		#cube.tvw2table()
		#cube.table_correctm2(corr=corr)
		#cube.mmttvw2clump()
		#cube.clump_distance()
		#cube.clump_mass()
		#cube.clump_gradient()
		#cube.table2image()
		###Merge Table
		if 0:
			cat=Catalogue().open(cube.cgrdtable)
			cat.write('clumps.cat','a' if i>0 else 'w')
		###Tile image
		if 0:
			pass



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
	def __init__(self, catfile='clumps.cat'):
		self.catfile = catfile
		self.cat = Catalogue().open(catfile)

	def _squareroot(img, vmin, vmax):
		#squareroot scale of image
		img = (img-vmin)/(vmax-vmin)
		img[img>1]=1
		img[img<0]=0
		img[np.isnan(img)]=1 #render nan as white
		img = img**0.5
		return img

	def _figuresetting(xspan=None, yspan=None, parts=1, separate=False, lv=False):
		#get figuresettings
		#ax factor must be the same for IntInt map
		if lv: axsize = (xspan/10/parts, yspan/100)
		else: axsize = (xspan/10/parts, yspan/10)
		marginleft = 0.35
		marginright = 0.25#0.25
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
		return figsize, figadjust


	def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list(
			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
			cmap(np.linspace(minval, maxval, n)))
		return new_cmap


	def plot_lbmap(self, figname, lrange, brange, fitsfile = '/Users/sz268601/Work/GuideMap/whole/tile_L_m0.fits', \
		parts=1, dpi=400):
		self.fitsfile = fitsfile
		hdu = fits.open(self.fitsfile)[0]
		img = PlotCatalogue._squareroot(hdu.data,0,18)
		ext = [*LinearWCS(hdu.header,1).extent, *(LinearWCS(hdu.header,2).extent)]
		lspan = max(lrange)-min(lrange)
		bspan = max(brange)-min(brange)

		plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
		plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True
		figsize, figadjust = PlotCatalogue._figuresetting(xspan=lspan,yspan=bspan, parts=parts, separate=False)
		fig,ax=plt.subplots(nrows=parts, figsize=figsize)
		if parts==1: ax=[ax]
		plt.subplots_adjust(**figadjust)
		cmap = PlotCatalogue._truncate_colormap(plt.get_cmap('gist_rainbow'), 0.8, 0.)
		for i,a in enumerate(ax):
			a.imshow(img, origin='lower', extent=ext, cmap='gray')
			a.set_aspect('equal')
			#a.plot(self.cat.l, self.cat.b, '.', markersize=0.02)
			im = a.scatter(self.cat.l, self.cat.b, marker='.', \
				c=self.cat.m1[:,1], cmap=cmap, vmin=-50,vmax=50, edgecolors='none', alpha=0.7, \
				s=np.sqrt(self.cat.sx*self.cat.sy)/5)
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
		#cb = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		cb = fig.colorbar(im, ax=ax.ravel().tolist())#cax=cbar_ax)
		cb.set_label(label='V$_{lsr}$ (km s$^{-1}$)', fontdict=dict(size=5))
		cb.ax.tick_params(axis='both', labelsize=5)
		#plt.show()
		plt.savefig('%s.png' % figname, dpi=dpi)


	def plot_plane(self, figname, dpi=400):
		plt.rcParams['xtick.top']=plt.rcParams['xtick.labeltop']=True
		plt.rcParams['ytick.right']=plt.rcParams['ytick.labelright']=True
		figsize, figadjust = PlotCatalogue._figuresetting(xspan=200, yspan=250, parts=1, separate=False)
		fig,ax=plt.subplots(figsize=figsize)
		plt.subplots_adjust(**figadjust)
		im = ax.scatter(self.cat.D*np.cos((self.cat.l-90)*np.pi/180), self.cat.D*np.sin((self.cat.l-90)*np.pi/180), marker='.', \
			c=np.log10(self.cat.mass), cmap='rainbow', vmin=0.5, vmax=4.5, edgecolors='none', alpha=0.7, \
			s=0.4)#np.log10(self.cat.sz))
		theta = np.linspace(0,np.pi,181)-np.pi/2
		ax.plot(8.5*np.cos(theta),8.5*np.sin(theta)-8.5)
		ax.plot(0.5*np.cos(theta*2),0.5*np.sin(theta*2)-8.5)
		ax.axis('equal')
		ax.set_xlim(-5,15)
		ax.set_ylim(-20,5)
		fig.colorbar(im, location='bottom',orientation='horizontal')
		plt.show()
		#plt.savefig('%s.png' % figname, dpi=dpi)



lrange=[19.75,122.25]
brange=[-5.25,5.25]
a=PlotCatalogue('clumps.cat')
#a.plot_lbmap('fig_lbmap_vs', lrange, brange, parts=2)
a.plot_plane('fig_plane_m')

'''
import matplotlib.pyplot as plt
from regulatetable import Catalogue, Sample
cat = Catalogue().open('clumps.cat')
#plt.scatter(np.sqrt(cat.sx*cat.sy), cat.m2[:,1], c=cat.m1[:,1], cmap='rainbow',s=0.2)
plt.scatter(cat.D*np.cos((cat.l-90)*np.pi/180), cat.D*np.sin((cat.l-90)*np.pi/180), c=np.log10(cat.m2[:,1]),\
	marker='.', s=0.01, cmap='rainbow',vmin=-0.5,vmax=-0.1)
#c=np.log10(cat.mass),
theta = np.linspace(0,np.pi,181)-np.pi/2
plt.plot(8.5*np.cos(theta),8.5*np.sin(theta)-8.5)
plt.plot(0.5*np.cos(theta*2),0.5*np.sin(theta*2)-8.5)
#plt.plot(cat.sz, cat.m2[:,1]*np.sqrt(3)/1e3,'.')
#plt.xscale('log')
#plt.yscale('log')
#plt.plot([.1,100],0.48*np.array([.1,100])**0.63)
plt.axis('equal')

plt.show()
'''


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


