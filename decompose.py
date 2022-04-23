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
#import matplotlib.ticker as ticker
#import matplotlib.colors as colors

#https://github.com/tvwenger/kd
from kd import rotcurve_kd
rotcurve = 'reid19_rotcurve' # the name of the script containing the rotation curve
dtor=np.pi/180
beam = 50#arcsec
R0 = 8.15

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
	SNR=2**np.linspace(1,6.5,40)
	for snr in tqdm(SNR):
		#upper = np.exp(np.exp(1.25-np.log(np.log(snr))*3))
		Sigma = 2**np.linspace(0,6,40)
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

def _Tex(Tpeak):
	#Nagahama et al. 1998
	return 5.53/np.log(1+5.53/(Tpeak+0.819))

def _thermal_velocity_dispersion(Tex):
	#this is thermal velocity dispersion of gas or SOUND SPEED(C_s)
	#VD_th,gas = sqrt(k_B*T/nu/m_p)
	#nu = 2.33 for mulecular gas / 1.27 for atomic gas
	return 0.05952026*np.sqrt(Tex)	#np.sqrt(con.k_B*u.K/2.33/con.m_p).to('km/s')

def _nonthermal_velocity_dispersion(velocity_dispersion_mol, Tex, w_mol=13+16):
	#this is nonthermal velocity dispersion
	#VD_nt^2 = VD_obs,i^2 - VD_th,i^2 (Hacar et al. 2022)
	#VD_th,i = sqrt(k_B*T/nu_i/m_p)
	#where i represent certain molecule, nu is molecular weight (nu_13CO = 29), m_p is proton mass
	thermal_velocity_dispersion_i = 0.09085373 * np.sqrt(Tex/w_mol) #np.sqrt(con.k_B*u.K/con.m_p).to('km/s')
	return np.sqrt(velocity_dispersion_mol**2 - thermal_velocity_dispersion_i**2)	#still need number of channels >5

def _total_gas_velocity_dispersion(nonthermal_velocity_dispersion, thermal_velocity_dispersion):
	return (nonthermal_velocity_dispersion**2+thermal_velocity_dispersion**2)

def _larson_distance(cat):
	###use Sigma*R sigma_v relation
	log_SR_far = np.log10(cat.SurfaceDensity*cat.angsz/2*cat.Dfar*1e3)		#in pc
	log_SR_near = np.log10(cat.SurfaceDensity*cat.angsz/2*cat.Dnear*1e3)	#in pc
	log_SR_larson = np.log10((cat.avm2/0.23)**(1/0.43))	#in pc

	###use R sigma_v relation
	#log_SR_far = np.log10(cat.angsz/2*cat.Dfar*1e3)		#in pc
	#log_SR_near = np.log10(cat.angsz/2*cat.Dnear*1e3)	#in pc
	#log_SR_larson = np.log10((cat.avm2/0.48)**(1/0.63))	#in pc
	
	#log_R_larson = np.log10((dv/0.778)**(1/0.43))	#in pc

	useD = cat.Dfar.copy()
	nanfar = np.isnan(cat.Dfar)
	useD[nanfar] = cat.Dnear[nanfar]

	usenear = np.isfinite(cat.Dnear) & np.isfinite(cat.Dfar)\
		& (np.abs(log_SR_far - log_SR_larson) > np.abs(log_SR_near - log_SR_larson))
	useD[usenear] = cat.Dnear[usenear]
	cat.D=useD 	#in kpc
	return cat


def m2correction_interpolate(filename='m2correction2sigma.dat', plot=False):
	#interpolate on the simulation points
	from scipy import interpolate
	Sigma, SNR, Fitting = np.loadtxt(filename).T
	#f = interpolate.interp2d(SNR, Fitting, Sigma, kind='linear')
	f = interpolate.LinearNDInterpolator(list(zip(SNR, Fitting)), Sigma/Fitting)#, fill_value=1)
	if plot:
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.scatter3D(SNR, Sigma, Fitting, c=Fitting, cmap='rainbow', marker='.')
		x = np.arange(2,81,0.25)
		y = np.arange(1,60,0.25)
		xx, yy = np.meshgrid(x, y)
		zz = f(xx,yy)
		#zz = interpolate.griddata(list(zip(SNR, Fitting)), Sigma, (xx, yy), method='nearest')
		#ax.contour3D(xx,yy,zz,levels=np.arange(1,20,0.5),cmap='rainbow')
		#ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
		ax.set_xlabel('SNR')
		ax.set_ylabel('$\sigma$');
		ax.set_zlabel('Moment2')
		plt.show()
	return f

def _c2v(header, c=None, unit='km/s'):
	if c is None: c=np.arange(header['NAXIS3'])
	vmps = (c-header['CRPIX3']+1) * header['CDELT3'] + header['CRVAL3']
	return vmps/1e3 if unit=='km/s' else vmps

def _v2c(header, v=None, unit='km/s'):
	if v is None: return np.arange(header['NAXIS3'])
	vmps = v*1e3 if unit=='km/s' else v
	return (vmps-header['CRVAL3'])/header['CDELT3']+header['CRVAL3']-1

class Decompose():
	if _user=='sz268601':
		fitspath = '/Users/sz268601/Work/DeepOutflow/procedure/prediction/whole'
		outpath = 'label/'
	else:
		fitspath = '/share/public/shbzhang/deepcube/'
		outpath = 'label/'
	overlap = 36
	boundary = ['229.167', '012.750', '-4.083', '+4.083']

	def __init__(self, prefix = '027.042+2.042', redo=True):
		#self._appendlog('Dealing with %s\n' % prefix)
		self.prefix = prefix
		self.redo = redo
		#input files
		self.usbfits = os.path.join(self.fitspath, prefix+'_U.fits')
		self.lsbfits = os.path.join(self.fitspath, prefix+'_L.fits')
		self.usbrmsfits = os.path.join(self.fitspath, prefix+'_U_rms.fits')
		self.lsbrmsfits = os.path.join(self.fitspath, prefix+'_L_rms.fits')
		#output files (each line is a pixel in a clump)
		self.dcpfits = os.path.join(self.outpath, prefix+'_decompose.fits')	#obsolete
		self.caafits = os.path.join(self.outpath, prefix+'_L_out.fits')	#cupid labelled cube
		self.mmtfits = os.path.join(self.outpath, prefix+'_L_mmt.fits')	#moment map for each clump
		self.tvwfits = os.path.join(self.outpath, prefix+'_U_tvw.fits')	#Temperature, velicity, width map for each clump
		self.mmttable = os.path.join(self.outpath, prefix+'_L_mmt.npy')
		self.mmtctable = os.path.join(self.outpath, prefix+'_L_mmtCM2.npy')
		self.tvwtable = os.path.join(self.outpath, prefix+'_U_tvw.npy')
		self.dsptable = os.path.join(self.outpath, prefix+'_nt.npy')
		self.dsttable = os.path.join(self.outpath, prefix+'_D.npy')	#distance copied from clump catalogue
		#clump table (each line is a clump)
		self.clumptable = os.path.join(self.outpath, prefix+'_clump.cat')	#geometry info for clumps
		self.cdsttable = os.path.join(self.outpath, prefix+'_clump_dst.cat')	#Distance info for clumps
		self.cmastable = os.path.join(self.outpath, prefix+'_clump_mas.cat')	#Mass/Sdensity info for clumps
		self.cgrdtable = os.path.join(self.outpath, prefix+'_clump_grd.cat')	#Gradient info for clumps
		self.avsparray = os.path.join(self.outpath, prefix+'_L_avsp.npy')	#AVerage SPectral array
		self.ccavsptable = os.path.join(self.outpath, prefix+'_clump_avsp.table')	
		self.ctnttable = os.path.join(self.outpath, prefix+'_clump_tnt.cat')	#Thermal / Non-Thermal info
		#fits map
		self.Dlimit = 1 #kpc
		self.ntmap = os.path.join(self.outpath, prefix+'_map_nt%1ikpc.fits' % self.Dlimit)	#Non-Thermal VD map
		self.machmap = os.path.join(self.outpath, prefix+'_map_mach%1ikpc.fits' % self.Dlimit)	#Non-Thermal VD map
		self.Nmap = os.path.join(self.outpath, prefix+'_map_N%1ikpc.fits' % self.Dlimit)	#Column density map
		self.ncompmap = os.path.join(self.outpath, prefix+'_map_ncomp%1ikpc.fits' % self.Dlimit)	#Number of Component map

		self.shape = [1, None, 281, 281]#fits.open(self.usbrmsfits)[0].data.shape
		self.xmin = 0 if prefix[:7]==self.boundary[0] else self.overlap//2
		self.xmax = self.shape[-1]-1 if prefix[:7]==self.boundary[1] else self.shape[-1]-1-self.overlap//2
		self.ymin = 0 if prefix[7:]==self.boundary[2] else self.overlap//2
		self.ymax = self.shape[-2]-1 if prefix[7:]==self.boundary[3] else self.shape[-2]-1-self.overlap//2

	def _appendlog(self, line):
		log = open('decompose.log','a')
		log.write('%s: %s\n' % (self.prefix,line))
		log.close()

	def _readytorun(inputs, output, redo=True):
		miss_inputs=False
		for file in inputs:
			if not exists(file):
				print('  Missing %s' % file)
				miss_inputs=True
		if miss_inputs: return False
		if redo: return True
		else:
			if exists(output):
				print('  Skip this for not REDO')
				return False
			else: return True


	##################################################################################################################
	def cube2caa(self):
		print("OBSOLETE, USE FellWalker")
		return
		#OBSOLETE, use fellwalker result now (cupid.py)
		if exists(self.lsbfits) & exists(lsbrmsfits):
			print('rms = %f' % self.lsbrms)
			#decompose 13CO datacube
			hdu = fits.open(self.lsbfits)[0]
			self.lsbrms = np.nanmean(fits.open(self.lsbrmsfits)[0].data) if exists(self.lsbrmsfits) else None
			label = Decompose._cube2lbl(hdu.data[0], min_intensity=3*self.lsbrms, intensity_step=3*self.lsbrms)	#remember to squeeze data
			fits.PrimaryHDU(data=label, header=hdu.header).writeto(self.dcpfits, overwrite=True)
			#filter labels that are small or weak
			###hdu = fits.open(self.lsbfits)[0]
			###label = fits.open(self.dcpfits)[0].data
			clabel = Decompose._lbl2caa(label, hdu.data[0], min_pixel=18, min_area=5, min_channel=5, min_peak=self.lsbrms*3)
			print('%i components found in total' % (clabel.max()+1))
			fits.PrimaryHDU(data=clabel, header=hdu.header).writeto(self.caafits, overwrite=True)

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
		print('Calculate moment from CAA')
		if Decompose._readytorun([self.lsbfits, self.caafits], self.mmtfits, self.redo):
			hdu = fits.open(self.lsbfits)[0]
			caa = fits.open(self.caafits)[0].data
			self.lsbrms = np.nanmean(fits.open(self.lsbrmsfits)[0].data) if exists(self.lsbrmsfits) else None
			mmt = Decompose._caa2mmt(caa[0], hdu.data[0], hdu.header, self.lsbrms)
			if len(mmt)>0:
				fits.PrimaryHDU(data=mmt, header=hdu.header).writeto(self.mmtfits, overwrite=True)

	def _caa2mmt(caa, intensity, header, rms):
		#find each component, calculate moment and stack into a cube
		dv = np.abs(header['CDELT3'])/1e3
		vaxis = _c2v(header)[:, np.newaxis, np.newaxis] #in km/s
		#vaxis = (np.arange(header['NAXIS3'])-header['CRPIX3']+1)*header['CDELT3']+header['CRVAL3']
		#vaxis = vaxis[:,np.newaxis,np.newaxis]/1e3	#to km/s
		nclump = np.nanmax(caa)
		moment=[]
		if np.isnan(nclump): return moment
		for l in tqdm(range(1,int(nclump)+1), desc='label_moment'):
			mask = caa==l
			cindex = np.argwhere(mask.any(axis=(1,2)))
			cslice = slice(cindex.min(), cindex.max()+1)
			#squeeze channels
			subvaxis = vaxis[cslice]
			submask = mask[cslice]
			subcube = intensity[cslice]*submask
			sumi = subcube.sum(axis=0, keepdims=True)
			m0 = sumi * dv																	#in K*km/s
			m1 = (subcube * subvaxis).sum(axis=0, keepdims=True) / sumi						#in km/s, moment 1 need at least 3 channels to be valid
			m2 = np.sqrt((subcube * (subvaxis-m1)**2).sum(axis=0, keepdims=True) / sumi)	#in km/s, moment 2 need at least 5 channels to be valid
			peak = subcube.max(axis=0, keepdims=True)	#in K
			nchan = submask.sum(axis=0, keepdims=True)	#in channel
			moment.append(np.vstack((m0,m1,m2,peak,nchan)))

		return np.array(moment)


	##################################################################################################################
	def mmt2tvw(self):
		#get Tpeak, velcity, and width for 12CO
		print('Calculate TVW from moment')
		if Decompose._readytorun([self.usbfits, self.mmtfits], self.tvwfits, self.redo):
			hdu = fits.open(self.usbfits)[0]
			mmt = fits.open(self.mmtfits)[0].data
			tvw = Decompose._mmt2tvw(mmt, hdu.data[0], hdu.header, width_factor=10)
			hdu.data = tvw
			fits.PrimaryHDU(data=tvw, header=hdu.header).writeto(self.tvwfits, overwrite=True)

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
					self._appendlog('Only %i channels in clump %i at (x=%i, y=%i)' % (moment[l,4,y,x],l,x,y))
		return tvw

	##################################################################################################################
	def mmt2table(self):
		print('Convert mmt to table')
		if Decompose._readytorun([self.mmtfits,], self.mmttable, self.redo):
			mmt = fits.open(self.mmtfits)[0].data
			mask = mmt[:,-1]>0
			table = Decompose._sparsearray2table(mmt, mask)
			np.save(self.mmttable, table)

	def tvw2table(self):
		print('Convert tvw to table')
		if Decompose._readytorun([self.tvwfits,], self.tvwtable, self.redo):
			mmt = fits.open(self.mmtfits)[0].data
			mask = mmt[:,-1]>0
			tvw = fits.open(self.tvwfits)[0].data
			table = Decompose._sparsearray2table(tvw, mask)
			np.save(self.tvwtable, table)

	def _sparsearray2table(array, mask):
		#convert sparse array to table
		idx = np.argwhere(mask)
		val = [array[:,i][mask] for i in range(array.shape[1])]
		return np.vstack((idx.T,np.array(val)))


	##################################################################################################################
	def caa2avsp(self):
		#calculate average spectra for each 13CO component
		print('Calculate average spectra from CAA')
		if Decompose._readytorun([self.lsbfits, self.caafits], self.avsparray, self.redo):
			hdu = fits.open(self.lsbfits)[0]
			caa = fits.open(self.caafits)[0].data
			avsp = Decompose._caa2avsp(caa[0], hdu.data[0])
			np.save(self.avsparray, avsp)

	def _caa2avsp(caa, intensity):
		#find each component, calculate moment and stack into a cube
		nclump = np.nanmax(caa)
		averspec=[]
		if np.isnan(nclump): return averspec
		for l in tqdm(range(1,int(nclump)+1), desc='label_moment'):
			mask = caa==l

			xindex = np.argwhere(mask.any(axis=(0,1)))
			xslice = slice(xindex.min(), xindex.max()+1)
			yindex = np.argwhere(mask.any(axis=(0,2)))
			yslice = slice(yindex.min(), yindex.max()+1)

			submask = mask[:,yslice,xslice]
			subcube = intensity[:,yslice,xslice]*submask

			npixel = submask.any(axis=0).sum()
			avsp = np.nansum(subcube, axis=(1,2)) / npixel
			averspec.append(avsp)
		return np.array(averspec)


	##################################################################################################################
	def table_correctm2(self, corr=lambda x,y: 1):
		#correct moment2 with m2correction simulation results
		print('Correct M2')
		doit = exists(self.mmttable) & exists(self.lsbrmsfits)
		if not self.redo: doit = doit & (not exists(self.mmtctable))
		if doit:
			mmt = np.load(self.mmttable)
			hdu = fits.open(self.lsbrmsfits)[0]
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
		print('Convert mmt&tvw table to clump table')
		if Decompose._readytorun([self.lsbfits, self.mmttable, self.tvwtable, self.usbrmsfits], self.clumptable, self.redo):
			header = fits.getheader(self.lsbfits)
			wcs = WCS(header, naxis=2)
			vaxis = _c2v(header)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			avsp = np.load(self.avsparray)

			cat = []
			for cn in range(int(mmtn.max())+1):
				cmask = mmtn==cn

				clm0 = mmtm0[cmask].sum()	#in
				cum0 = tvwm0[cmask].sum()

				#get position, remove the outsides
				cy = (mmtm0[cmask]*mmty[cmask]).sum()/clm0	#in array indices
				if (cy<self.ymin) | (cy>self.ymax): continue
				cx = (mmtm0[cmask]*mmtx[cmask]).sum()/clm0	#in array indices
				if (cx<self.xmin) | (cx>self.xmax): continue

				#get size
				cysz = np.sqrt((mmtm0[cmask]*(mmty[cmask]-cy)**2).sum()/clm0) #in pixel
				cxsz = np.sqrt((mmtm0[cmask]*(mmtx[cmask]-cx)**2).sum()/clm0) #in pixel

				#get moment 1
				cm1mask = cmask & (mmtnc>=3)
				mul = mmtm0[cm1mask] * mmtm1[cm1mask]
				clm1 = np.nansum(mul) / np.sum(mmtm0[cm1mask][np.isfinite(mul)])	#in km/s
				mul = tvwm0[cmask] * tvwm1[cmask]
				cum1 = np.nansum(mul) / np.sum(tvwm0[cmask][np.isfinite(mul)])	#in km/s

				#get moment 2
				cm2mask = cmask & (mmtnc>=5)
				mul = mmtm0[cm2mask] * mmtm2[cm2mask]
				clm2 = np.nansum(mul) / np.sum(mmtm0[cm2mask][np.isfinite(mul)])	#in km/s
				mul = tvwm0[cmask] * tvwm2[cmask]
				cum2 = np.nansum(mul) / np.sum(tvwm0[cmask][np.isfinite(mul)])	#in km/s

				#get peak
				clpk = mmtpk[cmask].max()	#in K
				cupk = tvwpk[cmask].max()	#in K

				#get area & volume
				carea = (mmtnc[cmask]>0).sum()	#in pixel
				cnc = mmtnc[cmask].sum()	#in voxel

				#get galactic lon/lat
				cl,cb = wcs.pixel_to_world_values(cx,cy)	#in deg
				###for l=180deg
				if np.isnan(cl):
					cl,cb = wcs.pixel_to_world_values(cx-43200,cy)
				if cl>229.75: continue
				if cl<11.75: continue

				#get averspec m1,m2
				sp = avsp[cn]
				smask = sp==0
				smask = smask & np.roll(smask,1) & np.roll(smask,-1)
				smask = smask | np.roll(smask,1) | np.roll(smask,-1)
				peakidx = np.argmax(sp)
				zeroidx = np.argwhere(smask)
				if (zeroidx<peakidx).any()>0:
					sp[:zeroidx[zeroidx<peakidx].max()]=0
				else: self._appendlog('clump out of velocity range of cube')
				if (zeroidx>peakidx).any()>0:
					sp[zeroidx[zeroidx>peakidx].min():]=0
				else: self._appendlog('clump out of velocity range of cube')
				sumi = sp.sum()
				cavm1 = (sp*vaxis).sum()/sumi
				cavm2 = np.sqrt((sp*(vaxis-cavm1)**2).sum() / sumi)

				cat.append({'prefix':self.prefix, 'num':cn, \
					'x':cx, 'y':cy, 'l':float(cl), 'b':float(cb), 'sx':cxsz, 'sy':cysz, \
					'area':carea, 'nc':cnc, 'peak':[cupk, clpk], \
					'm0':[cum0, clm0], 'm1':[cum1, clm1], 'm2':[cum2, clm2], \
					'avm1':cavm1, 'avm2':cavm2})
			cat = Catalogue(cat)
			cat.write(self.clumptable)
			return self.clumptable

	def clump2table(self, cat, key='D'):
		#copy clump parameters back to align with mmt table
		if Decompose._readytorun([self.mmttable, ], self.dsttable, self.redo):
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			subcat = cat[cat.prefix == self.prefix]
			mmtn = mmtn.astype(int)
			value = mmty.copy()
			value[:] = np.nan
			for c in subcat:
				idx = c.num == mmtn
				if idx.any(): value[idx]=c.D
			np.save(self.dsttable, np.array(value))


	##################################################################################################################
	def clump_distance(self):
		print('Add distance info to clump table')
		if Decompose._readytorun([self.clumptable], self.cdsttable, self.redo):
			cat = Catalogue().open(self.clumptable)
			if len(cat)==0:return
			###rotcurve_kd raise error if gl below and close to 90deg (eg. 89.99)
			###	use the average value with bias of +/-0.01 deg
			#try:
			#print(np.argwhere((cat.l>89.99)&(cat.l<90)))
			D = rotcurve_kd.rotcurve_kd(cat.l, cat.b, cat.avm1, rotcurve=rotcurve)
			'''
			###fill both distances of which are NAN
			prev_nannum = 0
			while True:
				idx = np.argwhere(np.isnan(D['near']) & np.isnan(D['far'])).ravel()
				if idx.size == prev_nannum:
					break
				prev_nannum = idx.size
				for i in idx:
					samecloud = (np.abs(cat.x-cat.x[i])<(cat.sx+cat.sx[i])*10) & (np.abs(cat.y-cat.y[i])<(cat.sy+cat.sy[i])*10) & (np.abs(cat.avm1-cat.avm1[i])<10)
					if np.isfinite(D['Rgal'][samecloud]).any():
						for key in D.keys():
							D[key][i] = np.nanmedian(D[key][samecloud])
			'''
			cat.Rgal = D['Rgal']
			cat.Dnear=D['near']
			cat.Dfar=D['far']

			cat.write(self.cdsttable)
			return self.cdsttable
			#except:
			#	print('Error in calculate distance for %s' % self.prefix)
			#	self._appendlog('Error in calculate distance for %s' % self.prefix)

			'''
			#P. Mroz et al 2018
			dtor=np.pi/180
			R0 = 8.5#kpc
			Vr = 233.6#-1.34*r
			V0 = 233.6-1.34*R0
			Vp = cat.avm1 / np.cos(cat.b*dtor)
			l=cat.l*dtor
			#Marc-Antoine et al. 2017
			factor = R0 * np.sin(l) / (Vp+V0*np.sin(l))
			Rgal = 233.6*factor / (1+1.34*factor)

			Dnear = R0 * np.cos(l) - np.sqrt(Rgal**2 - R0**2 * np.sin(l)**2)
			D = R0 * np.cos(l) + np.sqrt(Rgal**2 - R0**2 * np.sin(l)**2)
			#decide between near and far distance
			ang_sz = np.sqrt(np.sqrt((cat.sx*2.355*30)**2 - beam**2) * np.sqrt((cat.sy*2.355*30)**2 - beam**2))/3600*dtor #in rad
			dv = cat.avm2#*np.sqrt(3)	#3d vdisp in km/s
			log_R_larson = np.log10((dv/0.48)**(1/0.63))	#in pc
			#log_R_larson = np.log10((dv/0.778)**(1/0.43))	#in pc

			log_R_far = np.log10(ang_sz/2*D*1e3)
			log_R_near = np.log10(ang_sz/2*Dnear*1e3)
			usenear = (Rgal<R0) & (np.abs(log_R_far - log_R_larson) > np.abs(log_R_near - log_R_larson))
			D[usenear] = Dnear[usenear]
			SZ = ang_sz*D*1e3

			for i,(r,d,sz) in enumerate(zip(Rgal, D, SZ)):
				cat[i].Rgal = r 	#in kpc
				cat[i].D = d 		#in kpc
				cat[i].sz = sz 		#in pc
			cat.write(self.cdsttable)
			return self.cdsttable
			'''

	##################################################################################################################
	def clump_mass(self):
		print('Add mass info to clump table')
		if Decompose._readytorun([self.mmttable, self.tvwtable, self.cdsttable], self.cmastable, self.redo):
			cat = Catalogue().open(self.cdsttable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			#Nagahama et al. 1998
			Tex = _Tex(tvwpk)	#in K
			N = 1.49e20 / (1-np.exp(-5.29/Tex)) * mmtm0	#in cm-2
			for i,cn in enumerate(cat.num):
				cmask = (mmtn==cn)
				###N=1*u.Unit('cm-2')
				###D=1*u.Unit('kpc')
				#Tex
				cat[i].Tex = np.nanmax(Tex[cmask])	#in K
				#column density
				cat[i].N = np.nanmean(N[cmask]) 	#in cm-2
				####surface density
				###mean molecular weight per hydrogen molecule = 2.83 (Kauffmann et al. 2008)
				###f = (N*con.m_p*2.83).to('solMass/pc2') = 2.26662357e-20	
				cat[i].SurfaceDensity = cat[i].N*2.26662357e-20		#in Msun/pc2
				####convert factor from N(cm-2) to mass(M_sun) at D(kpc)
				###f = ((30/3600*np.pi/180*D)**2 * N * con.u*2.83 / con.M_sun).cgs = 4.76017541e-22
				cat[i].massperD2 = np.nansum(N[cmask])*4.76017541e-22#*cat[i].D**2	#in solar mass/D^2
			
			cat.angsz = np.sqrt(np.sqrt((cat.sx*2.355*30)**2 - beam**2) * np.sqrt((cat.sy*2.355*30)**2 - beam**2))/3600*dtor #in rad
			
			cat = _larson_distance(cat)

			cat.physz = cat.angsz * cat.D*1e3
			cat.mass = cat.massperD2 * cat.D**2

			####convert factor from mass(M_sun), Radius(pc) to density n(cm-3)
			#M=1*u.Unit('solMass')
			#R=1*u.Unit('pc')
			#f = (M / (con.m_p*2.83) / (4/3*np.pi*R**3)).cgs = 3.41335489

			cat.write(self.cmastable)
			return self.cmastable


	##################################################################################################################
	def table2dispersion(self):
		#calculate nonthermal velocity dispersion from mmt & tvw table
		print('Get NT info from mmt/tvw table')
		if Decompose._readytorun([self.mmttable,self.tvwtable], self.dsptable, self.redo):
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			#subtract thermal linewidth from m2
			Tex = _Tex(tvwpk)	#in K
			ThermalDispersion = _thermal_velocity_dispersion(Tex)	#np.sqrt(con.k_B/2.33/con.u*u.K).to('km/s')
			NonThermalDispersion12 = _nonthermal_velocity_dispersion(tvwm2, Tex, w_mol=12+16)
			NonThermalDispersion13 = _nonthermal_velocity_dispersion(mmtm2, Tex, w_mol=13+16)#still need number of channels >5
			dsp = np.c_[ThermalDispersion, NonThermalDispersion12, NonThermalDispersion13]
			np.save(self.dsptable, dsp.T)

	def clump_tnt(self):
		print('Add Thermal/NonThermal info to clump table')
		if Decompose._readytorun([self.mmttable, self.tvwtable, self.dsptable, self.cmastable], self.ctnttable, self.redo):
			cat = Catalogue().open(self.cmastable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			tdsp,nt0,nt1 = np.load(self.dsptable)
			def nanweimean(v,w):
				vw = v*w
				return np.nanmean(vw)/np.nanmean(np.isfinite(vw)*w)
			for i,cn in enumerate(cat.num):
				cmask = (mmtn==cn) & (mmtnc>=5)
				if cmask.any():
					#cat[i].tvd = np.nanmax(tdsp[cmask])
					cat[i].tvd = nanweimean(tdsp[cmask],mmtm0[cmask])
					#cat[i].ntvd = [np.nanmax(nt0[cmask]), np.nanmax(nt1[cmask])]
					cat[i].ntvd = [nanweimean(nt0[cmask],tvwm0[cmask]), \
						nanweimean(nt1[cmask],mmtm0[cmask])]
				else:
					cat[i].tvd = np.nan
					cat[i].ntvd = [np.nan, np.nan]
			cat.avntvd = _nonthermal_velocity_dispersion(cat.avm2, cat.Tex, w_mol=13+16)
			cat.write(self.ctnttable)
			return self.ctnttable


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	def dsptable2ntmap(self):
		#calculate nonthermal velocity dispersion from mmt & tvw table
		if Decompose._readytorun([self.mmttable, self.usbrmsfits, self.dsptable, self.dsttable], self.ntmap, self.redo):
			#cat = Catalogue().open(self.clumptable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			tdsp,nt0,nt1 = np.load(self.dsptable)
			tvwn,tvwy,tvwx,tvwm0,tvwm1,tvwm2,tvwpk = np.load(self.tvwtable)
			#Nagahama et al. 1998
			Tex = _Tex(tvwpk)	#in K
			N = 1.49e20 / (1-np.exp(-5.29/Tex)) * mmtm0	#in cm-2
			d = np.load(self.dsttable)
			header = fits.open(self.usbrmsfits, memmap=True)[0].header
			#ntimg = np.empty(self.shape[-2:], dtype=np.float32)
			#ntimg[:]=np.nan
			#machimg = np.empty(self.shape[-2:], dtype=np.float32)
			#machimg[:]=np.nan
			Nimg = np.empty(self.shape[-2:], dtype=np.float32)
			Nimg[:]=np.nan
			ncompimg = np.empty(self.shape[-2:], dtype=np.float32)
			ncompimg[:]=np.nan
			for x in range(self.shape[-1]):
				for y in range(self.shape[-2]):
					mask = (mmtx==x) & (mmty==y) & (mmtnc>=5) & np.isfinite(nt1) & np.isfinite(mmtm0) & (d<self.Dlimit)# (np.abs(mmtm1)<30)# & np.isfinite(d)
					if mask.any():
						###use D
						#idx = np.argmin(d[mask])
						###use V
						ntimg[y,x] = (nt1[mask]*mmtm0[mask]).sum() / mmtm0[mask].sum()
						#machimg[y,x] = (nt1[mask]/tdsp[mask]*mmtm0[mask]).sum() / mmtm0[mask].sum()
						Nimg[y,x] = np.nansum(N[mask])
						ncompimg[y,x] = mask.sum()
						'''
						idx = np.argmin(np.abs(mmtm1[mask]))	#index in the masked array
						#idx = np.argwhere(mask)[idx,0]	#index in original array
						nummask = mmtn==mmtn[mask][idx]
						idx = np.argmax(mmtm0[nummask])
						m1 = mmtm1[idx]
						m2 = mmtm2[idx]

						nearestv = mmtm1[idx]

						ntimg[y,x] = nt1[mask][idx]
						'''
			#fits.PrimaryHDU(data=ntimg, header=header).writeto(self.ntmap, overwrite=True)
			#fits.PrimaryHDU(data=machimg, header=header).writeto(self.machmap, overwrite=True)
			fits.PrimaryHDU(data=Nimg, header=header).writeto(self.Nmap, overwrite=True)
			fits.PrimaryHDU(data=ncompimg, header=header).writeto(self.ncompmap, overwrite=True)

	def dsptable2ncompmap(self):
		#calculate nonthermal velocity dispersion from mmt & tvw table
		if Decompose._readytorun([self.clumptable, self.mmttable, self.usbrmsfits], self.ntmap, self.redo):
			#cat = Catalogue().open(self.clumptable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			#d = np.array([cat.D[int(np.argwhere(cat.num==n))] if n in cat.num else np.inf for n in mmtn])
			header = fits.open(self.usbrmsfits, memmap=True)[0].header
			ncompimg = np.empty(self.shape[-2:], dtype=np.float32)
			ncompimg[:]=np.nan
			for x in range(self.shape[-1]):
				for y in range(self.shape[-2]):
					mask = (mmtx==x) & (mmty==y) & (mmtnc>=5) & np.isfinite(mmtm0) & (np.abs(mmtm1)<30)
					if mask.any():
						ncompimg[y,x] = mask.sum()
			fits.PrimaryHDU(data=ncompimg, header=header).writeto(self.ncompmap, overwrite=True)


	##################################################################################################################
	def clump_gradient(self):
		print('Add gradient info to clump table')
		if Decompose._readytorun([self.mmttable, self.cdsttable], self.cgrdtable, self.redo):
			cat = Catalogue().open(self.cdsttable)
			mmtn,mmty,mmtx,mmtm0,mmtm1,mmtm2,mmtpk,mmtnc = np.load(self.mmttable)
			for i,cn in enumerate(cat.num):
				cmask = (mmtn==cn)
				cmask = (mmtn==cn) & (mmtnc>3)
				x = mmtx[cmask]
				y = mmty[cmask]
				v = mmtm1[cmask]
				w = mmtm0[cmask]
				C = Decompose._gradient_fitting(x, y, v, w)	#unit in km/s/pixel
				GradientperDminus1 = [c/(30/3600*np.pi/180*1e3) for c in C]	#pixel to kpc
				#print(C)
				cat[i].GradientperDminus1 = GradientperDminus1
				#Gradient = GradientperDminus1 / D
			cat.write(self.cgrdtable)
			return self.cgrdtable

	def _gradient_fitting(x, y, v, w):
		A = np.c_[x*w, y*w, np.ones(x.shape)*w]
		C,R,_,E = lstsq(A, v*w)    # coefficients
		return list(C[0:2])	#gradient along x,y, Gtotal**2 = Gx**2+Gy**2

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	def orient_fitting(self):
		pass


	def clump_structurefunction2(self):
		print('')



if __name__ == '__main__':
	_tmp = Decompose()
	if _user=='sz268601':
		#usbfiles = ['027.042+2.042']
		usbfiles = glob.glob(os.path.join(_tmp.fitspath, '[012]*U_rms.fits'))#02[79]*+[24]*U_rms.fits'))
	else:
		usbfiles = glob.glob(os.path.join(_tmp.fitspath, '[012]*U.fits'))

	#m2correction_simulation()
	#corr = m2correction_interpolate('m2correction2sigma.dat', plot=True)
	prefixes = [os.path.basename(f)[:13] for f in usbfiles][:0]
	prefixes.sort()
	outcat=None
	#cat = Catalogue().open('clump_self0.10_equalusenear.cat')
	for i,prefix in enumerate(prefixes):
		print('>>>[%i/%i]%s<<<' % (i+1,len(prefixes), prefix))
		cube = Decompose(prefix=prefix, redo=True)
		###cube.cube2caa()
		#cube.caa2mmt()
		#cube.mmt2table()
		#cube.mmt2tvw()
		#cube.tvw2table()
		#cube.caa2avsp()
		#cube.table_correctm2(corr=corr)
		#outcat = cube.mmttvw2clump()

		#outcat = cube.clump_distance()
		#outcat = cube.clump_mass()
		#cube.clump2table(cat)

		#cube.table2dispersion()
		#outcat = cube.clump_tnt()
		#cube.dsptable2ntmap()

		#outcat = cube.clump_gradient()

	###Merge Clump Catalogue
	if 0:
		suffix = 'clump_tnt.cat'
		catfiles = glob.glob(os.path.join(_tmp.outpath, '*'+suffix))
		catfiles.sort()
		for i,catfile in enumerate(catfiles):
			cat=Catalogue().open(catfile)
			cat.write(suffix,'a' if i>0 else 'w')
	###Tile image
	if 0:
		suffix = 'mach3kpc.fits'
		fitsfiles = glob.glob(os.path.join(_tmp.outpath, '*'+suffix))
		import tile
		tile.tile(fitsfiles, suffix)
		
	###Extract parameters to npy file from Merged Clump Catalogue
	if 1:
		#distance
		useDcat = 'clump_self0.10_equalusenear.cat'
		cat=Catalogue().open(useDcat)
		D = cat.D
		np.save('D.npy', D)	#in kpc
		Rgal = cat.Rgal
		np.save('Rgal.npy', Rgal)	#in kpc
		np.save('Dnear.npy', cat.Dnear)
		np.save('Dfar.npy', cat.Dfar)
		np.save('Dorigin.npy', cat.Dorigin)

		#size
		np.save('l.npy', cat.l)	#in deg
		np.save('b.npy', cat.b)	#in deg
		angsz = cat.angsz
		np.save('angsz.npy', cat.angsz)	#in rad
		physz = cat.angsz*D*1e3
		np.save('physz.npy', physz)	#in pc
		R = cat.angsz*D*1e3/2
		np.save('R.npy', R)	#in pc
		area = cat.area
		np.save('area.npy', area)	#in pixel

		#N,mass,Tex
		cat=Catalogue().open('clump_mas.cat')
		np.save('Tex.npy', cat.Tex)	#in /cm2
		np.save('CD.npy', cat.N)	#in /cm2
		np.save('SD.npy', cat.SurfaceDensity)	#in Msun/pc2
		mass = cat.massperD2*D**2
		np.save('mass.npy', mass)	#in Msun
		n = mass/R**3*3.41335489	#in 1/cm3 (see comment in cube.clump_mass())
		np.save('n.npy', n)	#in pixel

		#velocity dispersion
		cat=Catalogue().open('clump_tnt.cat')
		np.save('avm1.npy', cat.avm1)	#in km/s
		np.save('avm2.npy', cat.avm2)	#in km/s
		np.save('tvd.npy', cat.tvd)	#in km/s
		np.save('ntvd.npy', cat.ntvd)	#in km/s
		np.save('avntvd.npy', cat.avntvd)	#in km/s
		np.save('Mach.npy', cat.avntvd/cat.tvd)

		cat=Catalogue().open('clump_grd.cat')
		gradient = cat.GradientperDminus1/D[:,np.newaxis]
		np.save('gradient.npy', gradient)	#in km/s/pc





	###DISTANCE DETERMINATION:
	#1.calculate kinematic distance from rotational curve with cube.clump_distance()
	#2.expolate distance for clump whose distance are both nan, put clumps whose Dnear>Dfar at tangent point with the next block
	#3.for clump Rgal<R0, determine in sz/dv/SD parameter space with the block after the next block


	###fill those clump whose distances are both nan with extrapolation.
	if 0:
		cat=Catalogue().open('clump_tnt.cat')
		origin = np.zeros(cat.size, dtype=int)

		prev_nannum = 0
		while True:
			unknown = np.isnan(cat.Dnear) & np.isnan(cat.Dfar)# & (cat.l>48) & (cat.l<55)
			idx = np.argwhere(unknown).ravel()
			if idx.size == prev_nannum:
				break
			prev_nannum = idx.size
			for i in tqdm(idx):
				samecloud = (np.abs(cat.l-cat.l[i])*60*2<(cat.sx+cat.sx[i])*2.355/2*10) & \
					(np.abs(cat.b-cat.b[i])*60*2<(cat.sy+cat.sy[i])*2.355/2*10) & \
					(np.abs(cat.avm1-cat.avm1[i])<10) & (~unknown)
				good = samecloud & np.isfinite(cat.Dnear)
				if good.sum()>=3:
					p = np.polyfit(cat[good].avm1, cat[good].Dnear, 1)
					cat[i].Dnear = cat[i].avm1*p[0]+p[1]
					if cat[i].Dnear<0: cat[i].Dnear = np.nan
					else: origin[i]=1
				good = samecloud & np.isfinite(cat.Dfar)
				if good.sum()>=3:
					p = np.polyfit(cat[good].avm1, cat[good].Dfar,1)
					cat[i].Dfar = cat[i].avm1*p[0]+p[1]
					if cat[i].Dfar<0: cat[i].Dfar = np.nan
					else: origin[i]=1
				good = samecloud & np.isfinite(cat.Rgal)
				if good.sum()>=3:
					p = np.polyfit(cat[good].avm1, cat[good].Rgal,1)
					cat[i].Rgal = cat[i].avm1*p[0]+p[1]
					origin[i]=1

		cat = _larson_distance(cat)

		#put clump with wrong distance at tangent point
		tangent = (cat.Dnear>cat.Dfar) & (cat.l<90)
		for i in np.argwhere(tangent).ravel():
			cat[i].Dnear = cat[i].Dfar = cat[i].D = R0 * np.cos(cat[i].l*dtor) / np.cos(cat[i].b*dtor)
			origin[i] = 2
		#original=0, filled=1, tangent=2
		cat.Dorigin = origin

		cat.physz = cat.angsz * cat.D *1e3	#in pc
		cat.mass = cat.massperD2*cat.D**2
		cat.write('clump_fillD.cat')

	#idx = (np.log10(cat.SurfaceDensity)<0.7) & (cat.avm2<1) & (cat.angsz>0.0008) & np.isfinite(cat.Dnear) & (cat.D==cat.Dfar) & (cat.D>8)


	###determine distance in sz/dv/SD log space according to clumps in outer galaxy.
	if 0:
		factor=0.10
		#for factor in [0.08,0.12]:
		print('Factor=%f' % factor)
		#for factor in [0.12, 0.15]:
		cat=Catalogue().open('clump_fillD.cat')

		cat.mass /= cat.D**2
		cat.physz = cat.angsz * cat.D *1e3	#in pc
		fixed = np.isfinite(cat.Dfar) & np.isnan(cat.Dnear) & (cat.Dorigin==0) & (np.abs(cat.l-180)>10)
		x1ref = np.log10(cat.physz[fixed])
		x2ref = np.log10(cat.avm2[fixed])
		x3ref = np.log10(cat.SurfaceDensity[fixed])
		#(cat.D>10) & (cat.Rgal>6.5) & (cat.Rgal<8.5) & (np.abs(cat.D*np.sin(cat.b*np.pi/180))>0.5) & (cat.SurfaceDensity>15)
		unknown = np.argwhere((cat.Dfar > cat.Dnear)).ravel()
		for i in tqdm(unknown):
			x1calnear = np.log10(cat.angsz[i]*cat.Dnear[i]*1e3)
			x1calfar = np.log10(cat.angsz[i]*cat.Dfar[i]*1e3)
			x2cal = np.log10(cat.avm2[i])
			x3cal = np.log10(cat.SurfaceDensity[i])
			### use 0.10/0.15 times 10% cut as binsize (np.percentile(X,90)-np.percentile(X,10))
			dense = (np.abs(x2ref-x2cal)<0.6295341622557098*factor) & (np.abs(x3ref-x3cal)<0.8743434266315732*factor)
			densenear = dense & (np.abs(x1ref-x1calnear)<1.2157975487515293*factor)
			densefar = dense & (np.abs(x1ref-x1calfar)<1.2157975487515293*factor)
			if densenear.sum() >= densefar.sum(): cat[i].D=cat[i].Dnear
			if densenear.sum() < densefar.sum(): cat[i].D=cat[i].Dfar
			#print(densenear.sum(), densefar.sum(), 'near' if densenear.sum() >= densefar.sum() else 'far',cat[i].b,cat[i].SurfaceDensity)

		cat.physz = cat.angsz * cat.D *1e3	#in pc
		cat.mass = cat.massperD2 * cat.D**2 	#in solar mass
		cat.write('clump_self%4.2f_equalusenear.cat' % factor)


