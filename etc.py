import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline

h = 6.626E-34 #Planck constant, units: J s
c = 3e8 #speed of light, units: m/s
kb = 1.380649e-23 #Boltzmann SI
sigma = 5.670374419E-08 #Stefan Boltzmann SI
Jy = 10**(-26) #1 Jansky, units: J s**-1 m**-2 Hz**-1

class PlanExposure:

	def __init__(self,paramfile=None, param_arr=None, integrate=True):
		self.integrate = integrate

		if paramfile:
			param = paramfile
			self.load_params(paramfile=param)

		elif isinstance(param_arr, np.ndarray):
			param = param_arr
			self.load_params(param_arr=param)

		else:
			print('Provide parameters in an input file ("paramfile") or in a 1d array ("param_arr").')

		self.get_usable_vals()
		self.calc_neff()


	def load_params(self, paramfile=None, param_arr=None):
		'''
		Parameters in paramfile: 

		Observing specifics
		---------------------------------------------------------------------------------------------------------------------------
		coadds: 						integer number
		frame time: 					in seconds
		sky_brightness: 				K-band sky brightness in units of Jansky per arcsec^2 (5e-3 for Palomar, 120e-6 for Dome C)
		
		Telescope properties/detector characteristics
		---------------------------------------------------------------------------------------------------------------------------
		focal_length:					in meters
		read_noise: 					in electrons per second
		dark_current: 					in electrons per second per pixel
		pix_size: 						pixel size in meters
		npix_edge:						Length of the edge of the detector array in pixels, FOV defined by npix_edge x npix_edge
		qe: 							Quantum efficiency of detector
		aperture_diam:					Aperture diameter in meters 

		Throughputs
		---------------------------------------------------------------------------------------------------------------------------
		avg_sky_transmiss: 				average sky transmission
		filter_throughput:  			filter efficiency
		transmiss_loss_per_surface: 	Loss in transmission after hitting each lens surface
		num_transmiss_losses: 			Number of lenses (2 corrector meniscii, 2 field flatteners = 4 in total)
		loss_per_reflection: 			Losses after each reflection
		num_reflections:				number of mirrors (1 primary mirror)
		
		K-dark observing
		---------------------------------------------------------------------------------------------------------------------------
		central_wavelength: 			Central wavelength in bandpass in meters 
		passband: 						K-dark wavelength range
		'''
		if paramfile:
			data = np.genfromtxt(paramfile, dtype=None,encoding=None)

		elif isinstance(param_arr, np.ndarray):
			data = param_arr

		self.coadds = float(data[0])
		self.frame_time = float(data[1])
		self.exptime = float(self.frame_time*self.coadds)
		self.sky_brightness = float(data[2])

		self.focal_length = float(data[3])
		self.read_noise = float(data[4])
		self.dark_current = float(data[5])
		self.pix_size = float(data[6])
		self.npix_edge = float(data[7])		
		self.qe = float(data[8]) #Only relevant if integrate=False 
		self.aperture_diam = float(data[9])

		self.avg_sky_transmiss = float(data[10]) #Only relevant if integrate=False 
		self.filter_throughput = float(data[11])
		self.transmiss_loss_per_surface = float(data[12])
		self.num_transmiss_losses = float(data[13])
		self.loss_per_reflection = float(data[14])
		self.num_reflections = float(data[15])

		self.lambda0 = float(data[16])
		self.passband = float(data[17])
		self.ambientT = float(data[18])

		# Only relevant if integrate=True
		self.qedata_file = str(data[19])
		self.skybkg_file = str(data[20])
		self.skytrans_file = str(data[21])


	def get_usable_vals(self,integrate=True):

		'''
		Get other usable quantities from the inputs
		'''

		#Conversions
		Rad_toArcsec = 206265 #1 radian = 206265 arcsec
		Arcsec_toRad = 1/Rad_toArcsec
		
		#Constants
		self.F0 = 3631*Jy #Zero point for AB system

		# Plate scale given the pixel size and focal length
		pix_size_rad = self.pix_size/self.focal_length #Pixel size in radians
		self.plate_scale = pix_size_rad * Rad_toArcsec #pix_size_rad/rad_toArcsec #Pixel size in arcsec

		# Bandwidth characteristics
		self.nu0 = c/self.lambda0 #Photon frequency at band center, units: Hz 
		self.E0_J = h*self.nu0 #Photon energy at band center, units: J
		self.delta_nu = c * ( (self.lambda0 - self.passband/2)**(-1)  - (self.lambda0 + self.passband/2)**(-1)) #Bandwidth, units: Hz

		# Collecting area
		#####################
		#Factor of 1.05 to roughly account for other obscurations (besides the physical size of detector package) such as from spider
		# To be modified
		beam_obstruction = (self.npix_edge*self.pix_size*1.05)**2 #Obscurations to the incident light, units: m**2
		#######################
		# Unobstructed area
		self.unobstructed_area = (np.pi*self.aperture_diam**2)/4 - beam_obstruction #Telescope's collecting area, units: m**2

		# Optical throughput
		self.optical_throughput = ((1-self.transmiss_loss_per_surface)**self.num_transmiss_losses)*((1-self.loss_per_reflection)**self.num_reflections)

		# Transmission
		# Fraction of photons at top of the atmosphere that make it to the detector
		
		if integrate:
			self.wavgrid = np.linspace(self.lambda0-self.passband/2, self.lambda0+self.passband/2)
			self.Egrid = h*c/self.wavgrid

			ashley_umwav0, ashley_log10uJysky =  np.loadtxt(self.skybkg_file,delimiter=',',unpack=True)
			ashley_wav0 = 1e-6*ashley_umwav0
			log10uJysky = np.interp(self.wavgrid, ashley_wav0, ashley_log10uJysky)
			self.skygrid = (10**log10uJysky)*1e-6 #Jy/arcsec^2

			ashley_umwav1, ashley_skytrans = np.loadtxt(self.skytrans_file,delimiter=',',unpack=True)
			self.skytransgrid = np.interp(self.wavgrid, ashley_umwav1*1e-6, ashley_skytrans)
			self.transmission_lambda = (self.skytransgrid*self.filter_throughput*self.optical_throughput)

			self.qewav_data, self.qe_data = np.loadtxt(self.qedata_file,delimiter=',',unpack=True)
			self.fit_qe = CubicSpline(self.qewav_data,self.qe_data)

			self.qe_lambda = np.zeros(self.wavgrid.size)
			detector_cutoff = self.qewav_data[-1]
			self.qe_lambda[np.where(self.wavgrid<=detector_cutoff)] = self.fit_qe(self.wavgrid[np.where(self.wavgrid<=detector_cutoff)])

		else:
			self.transmission = self.avg_sky_transmiss*self.filter_throughput*self.optical_throughput


	def calc_neff(self):
		'''
		For now, assuming that all light from a source falls on one pixel due to PSF undersampling. Will modify later
		'''
		self.neff = 1


	def calc_instr_noise_sq(self):
	    """
	    Combine all of the noise due to intrumentation into one value
	    """
	    instr_noise_sq =  self.neff* (self.coadds * (self.read_noise**2) + (self.dark_current * self.exptime)) #instrumental squared noise

	    return instr_noise_sq


	def calc_tse_noise(self):
		'''
		Calculates noise from the thermal self emission of the telescope, approximated to be dominated by window kept at ambient
		'''

		#MODIFY USING BLACKBODY EMISSION OF TELESCOPE: USE AN INPUT TEMPERATURE FOR TELESCOPE WALLS, INCORPORATE TEMP OF WINDOWS
		B_lambda = 2*h* (c**2)/(self.wavgrid**5 * (np.exp( h*c / (self.wavgrid * kb * self.ambientT)) - 1))
		dz = 109.735
		dc = 169.848
		theta_c = np.arccos(dz/dc) 
		F_lambda = np.pi*B_lambda*(np.sin(theta_c))**2 #Rybicki and Lightman eqn 1.13

		thickness=1.25 #cm
		transio2_wav_nm, transio2_1cm = np.loadtxt('./sio2_tran.csv',delimiter=',' , unpack=True)
		transio2_wav = transio2_wav_nm*1e-9
		transio2_1cm = transio2_1cm*1e-2
		transio2_1cm = np.interp(self.wavgrid, transio2_wav, transio2_1cm)

		transio2 = transio2_1cm**(thickness)
		tot_transmittance = transio2**2
		emissivity = 1-transio2

		factor_reflected = 0.7 #from ray tracing results from baffles

		A_pixel = (self.neff*self.pix_size)**2 
		Ltse = (1-factor_reflected)*(emissivity*F_lambda*A_pixel)  

		integrand = (Ltse/self.Egrid) * self.qe * self.filter_throughput

		tse_rate = trapz(integrand,self.wavgrid)

		return tse_rate


	def calc_sky_noise(self):
		'''
		Using the sky brightness in the input parameter txt file, calculates total sky noise [units of electrons/s]. 
		'''

		if self.integrate:
			Fsky_nu = ((self.skygrid*Jy)*(self.neff*self.plate_scale)**2) #Sky flux density, units: J s**-1 m**-2 Hz**-1
			Fsky_lambda = (c/(self.wavgrid**2)) * Fsky_nu #Sky flux density, units: J s**-1 m**-3

			integrand = (Fsky_lambda/self.Egrid) * self.qe_lambda * self.unobstructed_area * self.transmission_lambda #Units e-/(m*s)
			sky_rate =  trapz(integrand, self.wavgrid)  #Sky count rate at detector, units: e-/s 

		else:
			Fsky_nu = (self.sky_brightness*10**(-26))*(self.neff*self.plate_scale)**2  #J s**-1 m**-2 Hz**-1
			sky_rate = (Fsky_nu/self.E0_J) * self.qe * self.unobstructed_area * self.transmission * self.delta_nu  #Sky count rate at detector, units: e-/s 

		return sky_rate


	def calc_src_rate(self,AB_magnitude):
		'''
		Calculates source count rate [units of electrons/s] given an AB magnitude. 
		'''
		Fsrc_nu = self.F0 * 10**(-0.4*AB_magnitude) #Source flux density, units: J s**-1 m**-2 Hz**-1

		if self.integrate:
			Fsrc_lambda = (c/(self.lambda0**2)) * Fsrc_nu #Source flux density, units: J s**-1 m**-3
			integrand = (Fsrc_lambda/self.E0_J) * self.qe_lambda * self.unobstructed_area * self.transmission_lambda #Units e-/(m*s)
			source_rate = trapz(integrand, self.wavgrid) #Source count rate at detector, units: e-/s	

		else:
			source_rate = (Fsrc_nu/self.E0_J) * self.qe * self.unobstructed_area * self.transmission * self.delta_nu #Source count rate at detector, units: e-/s
		
		return source_rate


	def calc_snr(self,AB_magnitude):
		'''
		Calculates SNR given an AB magnitude for a source
		'''
		source_rate = self.calc_src_rate(AB_magnitude) 
		sky_rate = self.calc_sky_noise()
		instr_noise_sq = self.calc_instr_noise_sq()
		if self.integrate:
			tse_rate = self.calc_tse_noise()
		else:
			tse_rate=0

		source_counts = source_rate * self.exptime
		sky_counts = sky_rate * self.exptime
		tse_counts = tse_rate * self.exptime

		#SNR = source_counts/np.sqrt(source_counts+sky_counts+instr_noise_sq)
		SNR = source_counts/np.sqrt(source_counts+sky_counts+tse_counts+instr_noise_sq)

		data = np.array([source_rate, sky_rate, tse_rate, instr_noise_sq, SNR])

		return data


	def plot_snr(self,mag_range):
		'''
		Plot SNR for a source of a given magnitude, given some predefined exposure time included in etc_param.txt
		'''
		snrlist = []
		srcrate_list = []

		for m in mag_range:
			data = self.calc_snr(m)
			snrlist.append(data[-1])
			srcrate_list.append(data[0])

		snrs= np.array(snrlist)
		srcrates= np.array(srcrate_list)

		fig, ax1 = plt.subplots()
		ax1.plot(mag_range, snrs)
		ax1.set_xlabel('Magnitude')
		ax1.set_ylabel('SNR')
		ax1.tick_params(axis='y')
		ax1.set_yscale('log')

		return srcrates, snrs


	def plot_exptime(self,mag_range,snr_desired):
		'''
		Solve for exposure time needed to reach a given SNR 
		'''
		D = self.dark_current
		R = self.neff*self.coadds*(self.read_noise**2) 
		B = self.neff*self.calc_sky_noise()

		times = []

		for m in mag_range:
			S = self.calc_src_rate(m)
			# Solving quadratic: S**2 * texp**2 - snr_desired**2 * (S+B+D)*texp - (snr_desired**2+R) = 0
			coeff = [S**2, -snr_desired**2 * (S+B+D), -(snr_desired**2+R)]
			texp = np.roots(coeff)
			times.append(texp[texp>0])


		fig, ax1 = plt.subplots()
		ax1.plot(mag_range, times)
		ax1.set_xlabel('Magnitude')
		ax1.set_ylabel('Required Time (s)')
		ax1.tick_params(axis='y')
		ax1.set_yscale('log')

		return times






