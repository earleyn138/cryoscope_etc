import numpy as np
import matplotlib.pyplot as plt

class PlanExposure:

	def __init__(self,paramfile):
		self.load_params(paramfile)
		self.get_usable_vals()
		self.calc_neff()


	def load_params(self, paramfile):
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

		data = np.genfromtxt(paramfile, delimiter='\t',usecols=0)

		self.coadds = data[0]
		self.frame_time = data[1]
		self.exptime = self.frame_time*self.coadds
		self.sky_brightness = data[2]

		self.focal_length = data[3]
		self.read_noise = data[4] 
		self.dark_current = data[5]
		self.pix_size = data[6]
		self.npix_edge = data[7]		
		self.qe = data[8]
		self.aperture_diam = data[9] 

		self.avg_sky_transmiss = data[10]
		self.filter_throughput = data[11]
		self.transmiss_loss_per_surface = data[12]
		self.num_transmiss_losses = data[13]
		self.loss_per_reflection = data[14]
		self.num_reflections = data[15]

		self.lambda0=data[16]
		self.passband = data[17]


	def get_usable_vals(self):

		'''
		Get other usable quantities from the inputs
		'''

		#Conversions
		Rad_toArcsec = 206265 #1 radian = 206265 arcsec
		Arcsec_toRad = 1/Rad_toArcsec
		
		#Constants
		h = 6.626E-34 #Planck constant, units: J s
		c = 3e8 #speed of light, units: m/s
		Jy = 10**(-26) #1 Jansky, units: J s**-1 m**-2 Hz**-1
		self.F0 = 3631*Jy #Zero point for AB system

		# Plate scale given the pixel size and focal length
		pix_size_rad = self.pix_size/self.focal_length #Pixel size in radians
		self.plate_scale = pix_size_rad * Rad_toArcsec #7.1 #pix_size_rad/rad_toArcsec #Pixel size in arcsec

		# Bandwidth characteristics
		nu0 = c/self.lambda0 #Photon frequency at band center, units: Hz 
		self.E0_J = h*nu0 #Photon energy at band center, units: J
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


	def calc_sky_noise(self):
		'''
		Using the sky brightness in the input parameter txt file, calculates total sky noise [units of electrons/s]. 
		'''
		Fsky = (self.sky_brightness*10**(-26))*(self.neff*self.plate_scale)**2  #J s**-1 m**-2 Hz**-1
		sky_rate = (Fsky/self.E0_J) * self.qe * self.unobstructed_area * self.delta_nu * self.transmission#Sky count rate at detector, units: e-/s 

		return sky_rate


	def calc_src_rate(self,AB_magnitude):
		'''
		Calculates source count rate [units of electrons/s] given an AB magnitude. 
		'''
		Fnu = self.F0 * 10**(-0.4*AB_magnitude) #Source flux density, units: J s**-1 m**-2 Hz**-1
		source_rate = (Fnu/self.E0_J) * self.qe * self.unobstructed_area * self.delta_nu * self.transmission #Source count rate at detector, units: e-/s
		return source_rate


	def calc_snr(self,AB_magnitude):
		'''
		Calculates SNR given an AB magnitude for a source
		'''
		source_rate = self.calc_src_rate(AB_magnitude) 
		sky_rate = self.calc_sky_noise()
		instr_noise_sq = self.calc_instr_noise_sq()

		source_counts = source_rate * self.exptime
		sky_counts = sky_rate * self.exptime

		SNR = source_counts/np.sqrt(source_counts+sky_counts+instr_noise_sq)

		data = np.array([source_rate, sky_rate, instr_noise_sq])

		return data, SNR


	def plot_snr(self,mag_range):
		'''
		Plot SNR for a source of a given magnitude, given some predefined exposure time included in etc_param.txt
		'''
		snrlist = []
		srcrate_list = []

		for m in mag_range:
			data, snr = self.calc_snr(m)
			snrlist.append(snr)
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






