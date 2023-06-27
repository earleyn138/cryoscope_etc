
import numpy as np

class PhotoParameters:

	def __init__(self, paramfile):

		data = np.genfromtxt(paramfile, delimiter='\t',usecols=0)

		'''
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
		plate_scale: 					arcsec per pixel
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


		self.central_wavelength=data[16]
		self.passband = data[17]

		self.get_usable_quantities()


		
	def get_usable_quantities(self):

		#Conversions:
		Jy_toSI = 1E-26 #1 Jy = 1E-26 W/m^2/Hz
		Rad_toArcsec = 206265 #1 radian = 206265 arcsec
		Arcsec_toRad = 1/Rad_toArcsec
		
		#Constants
		h = 6.626E-34 #Joule sec
		c = 3e8 # speed of light in m/s
		self.vega_0mag = 3631 #Jy , AB zero point

		# Converting inputs in usable quantities
		pix_size_rad = self.pix_size/self.focal_length #Pixel size in radians
		self.plate_scale = pix_size_rad * Rad_toArcsec #7.1 #4.13 #pix_size_rad/rad_toArcsec #Pixel size in arcsec

		central_freq = c/self.central_wavelength #frequency of photon in Hz at band center

		#Energy of photon at band center
		E0_J = h*central_freq 

		#Converting the bandwidth in terms of wavelength to be in terms of frequency 
		frequency_bandwidth = self.passband*(central_freq**2)/c

		#####################
		#Factor of 1.05 to roughly account for other obscurations (besides the physical size of detector package) such as from spider
		# To be modified
		beam_obstruction = (self.npix_edge*self.pix_size*1.05)**2
		#######################

		# Unobstructed area
		self.unobstructed_area = (np.pi*self.aperture_diam**2)/4 - beam_obstruction

		# Optical throughpout
		self.optical_throughput = ((1-self.transmiss_loss_per_surface)**self.num_transmiss_losses)*((1-self.loss_per_reflection)**self.num_reflections)

		# Transmission
		# Calculates fraction of photons at top of the atmosphere that make it to the detector
		self.transmission = self.avg_sky_transmiss*self.filter_throughput*self.optical_throughput*self.unobstructed_area


		# Converting energy flux density to in-band photon flux, units = ph/s/m^2
		self.Jy_toIBPhotonFlux = (Jy_toSI/E0_J) * frequency_bandwidth







