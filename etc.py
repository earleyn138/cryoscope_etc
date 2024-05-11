import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

# Fundamental constants and conversions
h = 6.626E-34  # Planck constant, units: J s
c = 3e8  # speed of light, units: m/s
kb = 1.380649e-23  # Boltzmann SI
sigma = 5.670374419E-08  # Stefan Boltzmann SI
Jy = 10**(-26)  # 1 Jansky, units: J s**-1 m**-2 Hz**-1
Rad_toArcsec = 206265  # 1 radian = 206265 arcsec
Arcsec_toRad = 1 / Rad_toArcsec
F0 = 3631 * Jy  # Zero point for AB system


class PlanExposure:
	"""
	A class incorporating telescope and on-site properties that can be used to calculate the signal-to-noise ratio (SNR)
	for given point source magnitudes and observing procedures.

	Attributes
	---------------------------------------------------------------------------------------------------------------------------------------------------
	Telescope, on-site, observing properties from user input:
		OBSERVING SPECIFICS
			coadds: int
				Number of coadds
			frame_time: float
				Time for one frame (s)
			avg_skyBrightness: float
				K-band sky brightness in units of Jansky per arcsec^2
				(5e-3 for Palomar, 120e-6 for Dome C), (relevant only if sky spectrum is not provided)
			focal_length: float
				Focal length of the telescope (m)

		TELESCOPE PROPERTIES/DETECTOR CHARACTERISTICS
			read_noise: float
				Read noise in electrons per second
			dark_current: float
				Dark current in electrons per second per pixel
			pixel_size: float
				Pixel size in meters
			beam_obstruction: float
				Beam obstruction in m^2
			avg_qe: float
				Average quantum efficiency of detector (if no data file provided for QE curve)
			aperture_diam: float
				Aperture diameter in meters

		THROUGHPUTS INFO
			avg_skyTransmission: float
				Average sky transmission
			filter_throughput: float
				Filter efficiency
			transmiss_loss_per_surface: float
				Loss in transmission after hitting each lens surface
			num_transmiss_losses: int
				Number of lenses (2 corrector meniscii, 2 field flatteners = 4 in total)
			loss_per_reflection: float
				Losses after each reflection
			num_reflections: int
				Number of mirrors (1 primary mirror)

		K-DARK OBSERVING
			lambda0: float
				Central wavelength of bandpass in meters
			passband: float
				Wavelength range (if filterTransmission_file not provided)
			ambientT: float
				Temperature of the first corrector (the window). Specify in units of Celsius

		OPTIONAL DATA FILES (to integrate over)
			skyBrightness_file: str
				Sky brightness spectral file
			skyTransmission_file: str
				Sky transmission spectral file
			qe_file: str
				QE spectral file
			filterTransmission_file: str
				Filter transmission spectral file
			sio2Transmission_file: str
				SiO2 transmission data for thermal self emission (TSE) calculations

	Quantities generated from the inputs:
		plate_scale: float
			Plate scale in arcsec/pixel. Calculated from pixel size and focal length.
		nu0: float
			Photon frequency at band center, units: Hz
		E0_J: float
			Photon energy at band center, units: J
		delta_nu: float
			Bandwidth, units: Hz
		unobstructed_area: float
			Telescope's collecting area, units: m^2
		optical_throughput: float
			Optical throughput
		transmission: float
			Fraction of photons at top of the atmosphere that make it to the detector
		neff: int
			Effective number of pixels in aperture
		wavgrid: np.ndarray
			Grid of wavelengths for integration
		Egrid: np.ndarray 
			Grid of photon energies for integration
		skyB_grid: np.ndarray
			Sky brightness spectrum in units of Jy/arcsec^2
		skyT_grid: np.ndarray
			Sky transmission spectrum
		qe_lambda: np.ndarray
			Quantum efficiency as a function of wavelength
		transmission_lambda: np.ndarray
			Transmission as a function of wavelength
	---------------------------------------------------------------------------------------------------------------------------------------------------

	Methods
	---------------------------------------------------------------------------------------------------------------------------------------------------
	load_params(data)
		Load parameters from input data
	get_usable_vals()
		Get other usable quantities from the inputs
	calc_neff()
		Calculate the effective number of pixels in the aperture
	calc_detector_counts()
		Calculates electrons from the detector noise in 1 frame. Combines dark current and read noise
	calc_tse_rate()
		Calculates noise from the thermal self emission of the telescope if integrating over spectral data
	calc_sky_rate()
		Using the sky brightness in the input parameter txt file, calculates total sky noise [units of electrons/s]
	calc_src_rate(ab_mag)
		Calculates source count rate [units of electrons/s] given an AB magnitude
	calc_snr_frame(ab_mag,plot)
		Calculates SNR for a single frame given an AB magnitude for a source
	calc_snr(ab_mag,plot)
		Calculates integrated SNR for an observation given an AB magnitude for a source.
		Option to plot SNR vs. magnitude if plot=True
	calc_time(mag_range,snr_desired,plot)
		Solve for exposure time needed to reach a given SNR. Option to plot exposure time vs. magnitude if plot=True
	---------------------------------------------------------------------------------------------------------------------------------------------------
	"""

	def __init__(self, param_file=None, param_arr=None, integrate=True):
		"""
		Initializes the PlanExposure class with the given parameters

		Parameters
		---------------------------------------------------------------------------------------------------------------------------------------------------
		:param param_file: str (optional, if param_arr specified) -
			Path to a text file containing the parameters for the telescope and observation.
			Either param_file or param_arr must be specified.
		:param param_arr: np.ndarray (optional, if param_file specified) -
			Array containing the parameters for the telescope and observation.
			Either param_file or param_arr must be specified.
		:param integrate: bool (optional) -
			By default: True, integrate over spectral data.
			If False, use average values for throughput and quantum efficiency
		---------------------------------------------------------------------------------------------------------------------------------------------------
		"""
		# Initialize attributes
		self.neff = None
		self.transmission = None
		self.transmission_lambda = None
		self.qe_lambda = None
		self.skyT_grid = None
		self.skyB_grid = None
		self.Egrid = None
		self.wavgrid = None
		self.optical_throughput = None
		self.unobstructed_area = None
		self.delta_nu = None
		self.E0_J = None
		self.nu0 = None
		self.plate_scale = None
		self.frame_time = None
		self.avg_skyBrightness = None
		self.focal_length = None
		self.read_noise = None
		self.dark_current = None
		self.pixel_size = None
		self.beam_obstruction = None
		self.avg_qe = None
		self.aperture_diam = None
		self.avg_skyTransmission = None
		self.filter_throughput = None
		self.transmiss_loss_per_surface = None
		self.num_transmiss_losses = None
		self.loss_per_reflection = None
		self.num_reflections = None
		self.lambda0 = None
		self.passband = None
		self.ambientT = None
		self.skyBrightness_file = None
		self.skyTransmission_file = None
		self.qe_file = None
		self.filterTransmission_file = None
		self.sio2Transmission_file = None

		# Set integrate to True by default
		self.integrate = integrate

		# Load parameters from input file or array
		if param_file:
			data = np.genfromtxt(param_file, dtype=None, encoding=None)
			self.load_params(data)

		elif isinstance(param_arr, np.ndarray):
			data = param_arr
			self.load_params(data)

		# If no parameters provided, raise an error
		else:
			raise ValueError('Provide parameters in an input file ("param_file") or in a 1d array ("param_arr").')

		# Get other usable values from the inputs
		self.get_usable_vals()
		# Calculate effective number of pixels for a source.
		# This is set to 1 to account for PSF undersampling expected for diffraction limited performance
		self.calc_neff()

	def load_params(self, data):
		"""
		Load parameters from input data

		:param data: np.ndarray - Array containing the parameters for the telescope and observation
		"""

		self.frame_time = float(data[0])
		self.avg_skyBrightness = float(data[1])
		self.focal_length = float(data[2])
		self.read_noise = float(data[3])
		self.dark_current = float(data[4])
		self.pixel_size = float(data[5])
		self.beam_obstruction = float(data[6])
		self.avg_qe = float(data[7])  # Only relevant if integrate=False
		self.aperture_diam = float(data[8])

		self.avg_skyTransmission = float(data[9])  # Only relevant if integrate=False
		self.filter_throughput = float(data[10])
		self.transmiss_loss_per_surface = float(data[11])
		self.num_transmiss_losses = float(data[12])
		self.loss_per_reflection = float(data[13])
		self.num_reflections = float(data[14])

		self.lambda0 = float(data[15])
		self.passband = float(data[16])
		self.ambientT = 273+float(data[17])  # units of K

		# Only relevant if integrate=True
		self.skyBrightness_file = str(data[18])
		self.skyTransmission_file = str(data[19])
		self.qe_file = str(data[20])
		self.filterTransmission_file = str(data[21])
		self.sio2Transmission_file = str(data[22])

	def get_usable_vals(self):
		"""
		Get other usable physical quantities from the inputs
		"""

		# Plate scale given the pixel size and focal length
		pixel_size_rad = self.pixel_size/self.focal_length  # Pixel size in radians
		self.plate_scale = pixel_size_rad * Rad_toArcsec  # Pixel size in arcsec

		# Bandwidth characteristics
		self.nu0 = c/self.lambda0  # Photon frequency at band center, units: Hz
		self.E0_J = h*self.nu0  # Photon energy at band center, units: J
		self.delta_nu = (
				c * ((self.lambda0 - self.passband/2)**(-1) - (self.lambda0 + self.passband/2)**(-1))
		)  # Bandwidth, units: Hz

		# Collecting area
		self.unobstructed_area = (
				(np.pi*self.aperture_diam**2)/4 - self.beam_obstruction
		)  # Telescope's collecting area, units: m**2

		# Optical throughput
		transmission_losses = (1-self.transmiss_loss_per_surface)**self.num_transmiss_losses
		reflection_losses = (1-self.loss_per_reflection)**self.num_reflections
		self.optical_throughput = transmission_losses * reflection_losses

		# Transmission, fraction of photons at top of the atmosphere that make it to the detector, 
		# not accounting for QE and unobstructed area yet.
		# If integrating, load the data files and create a grid of wavelengths over which to integrate. 
		if self.integrate:
			if not self.filterTransmission_file:
				# If not integrating over filter transmission data, assume a flat transmission specified by 
				# user-input filter_throughput. Create a grid of wavelengths specified by lambda0 and passband.
				self.wavgrid = np.linspace(self.lambda0-self.passband/2, self.lambda0+self.passband/2)
			else:
				# Otherwise, load the filter transmission data. This overrides the user-input filter_throughput.
				# Grid of wavelengths set by those in the filterTransmission_file.
				wavgrid_nm, self.filter_throughput = np.loadtxt(
					self.filterTransmission_file, delimiter=',', unpack=True, skiprows=1
				)
				self.wavgrid = wavgrid_nm[::-1]*(10**-9)  # wavgrid in units of m
				self.filter_throughput = self.filter_throughput[::-1]/100 

			self.Egrid = h*c/self.wavgrid  # Grid of photon energies for integration

			# Sky brightness spectrum in units of Jy/arcsec^2
			skyb_um, skyb_log = np.loadtxt(
				self.skyBrightness_file, delimiter=',', unpack=True
			)  # Sky spectrum is in units of log10(uJy/arcsec^2). Wavelengths are in microns.
			sky_wav = 1e-6*skyb_um
			skyb_log = np.interp(self.wavgrid, sky_wav, skyb_log)  # Interpolating to the grid of wavelengths
			self.skyB_grid = (10 ** skyb_log) * 1e-6  # units Jy/arcsec^2

			# Sky transmission spectrum
			skyt_um, skyt = np.loadtxt(self.skyTransmission_file, delimiter=',', unpack=True)
			skyt_wav = 1e-6*skyt_um
			self.skyT_grid = np.interp(self.wavgrid, skyt_wav, skyt)

			# Quantum efficiency as a function of lambda. Not included in transmission yet. 
			qedata_wav, qedata = np.loadtxt(self.qe_file, delimiter=',', unpack=True)
			self.qe_lambda = np.interp(self.wavgrid, qedata_wav, qedata)

			# Transmission as a function of lambda
			self.transmission_lambda = (self.skyT_grid*self.filter_throughput*self.optical_throughput)
			
		# Otherwise, use average values for the filter throughput and sky Transmission.
		else:
			self.transmission = self.avg_skyTransmission*self.filter_throughput*self.optical_throughput

	def calc_neff(self):
		"""
		Calculate the effective number of pixels for a point source.
		For now, assuming that all light from a source falls on one pixel due to expected PSF undersampling. 
		"""
		# Cryoscope is diffraction-limited in the K-dark bandpass. PSF will be undersampled.
		res = 1.22 * self.lambda0 / self.aperture_diam  # Angular resolution in radians
		neff = res/(self.plate_scale*Arcsec_toRad)
		if neff < 1:
			self.neff = 1

	def calc_detector_counts(self):
		"""
		Calculates electrons from the detector noise in 1 frame. Combines dark current and read noise.

		:return: detector_noise: float - Number of electrons from dark current and read noise in a single frame
		"""

		detector_counts = self.neff*((self.read_noise**2) + self.dark_current*self.frame_time)

		return detector_counts

	def calc_tse_rate(self):
		"""
		Calculates electrons/s from the thermal self emission of the telescope if integrating over spectral data,
		approximated to be dominated by window kept at ambient. Otherwise, set to zero.
		Only available for prototype (Cryoscope Pathfinder) parameters as of now.
		
		:return: tse_rate: float - Thermal self emission rate [units of electrons/s]
		"""
		# Telescope properties for TSE calculations
		thickness = 1.25  # Thickness of the window in cm
		dz = 109.735  # Distance from the center of detector to the center of pupil stop
		dc = 169.848  # Distance from the center of detector to the edge of pupil stop.
		factor_reflected = 0.7  # from ray tracing results. 70% of window TSE is reflected back due to baffles

		if self.integrate and self.sio2Transmission_file != 'Skip':
			# Calculating emissivity of the SiO2 window.
			sio2_tran_data = np.loadtxt(
				self.sio2Transmission_file, delimiter=',').T  # Load in the SiO2 transmission data for 1cm thick sample
			sio2_tran_1cm = sio2_tran_data[1]/100
			sio2_tran = sio2_tran_1cm ** thickness  # Transmission for the thickness of the window
			tot_transmittance = sio2_tran ** 2  # Total transmission through the windows
			emissivity = np.mean(1 - tot_transmittance)  # Emissivity of the windows

			# Calculating the flux density from the thermal self emission of the window
			b_lambda = 2*h*(c**2)/(
					self.wavgrid**5 * (np.exp(h*c / (self.wavgrid * kb * self.ambientT)) - 1)
			)  # Planck's law
			theta_c = np.arccos(dz/dc)
			f_lambda = np.pi*b_lambda*(np.sin(theta_c))**2  # Rybicki and Lightman eqn 1.13

			# Calculating the power emitted by the window to the central detector pixel
			pixel_area = (self.neff*self.pixel_size)**2
			lum_tse = (1-factor_reflected)*(emissivity*f_lambda*pixel_area)

			# Integrating to get the TSE rate
			integrand = (lum_tse/self.Egrid) * self.qe_lambda * self.filter_throughput
			tse_rate = trapz(integrand, self.wavgrid)

		else:
			tse_rate = 0

		return tse_rate

	def calc_sky_rate(self):
		"""
		Using the sky brightness in the input parameter txt file, calculates total sky rate [units of electrons/s].
		
		:return: sky_rate: float - Sky count rate at detector, units: e-/s
		"""

		if self.integrate:
			sky_flux_nu = (
					(self.skyB_grid * Jy) * (self.neff * self.plate_scale) ** 2
			)  # Sky flux density, units: J s**-1 m**-2 Hz**-1
			sky_flux_lambda = (
					(c/(self.wavgrid**2)) * sky_flux_nu 
			)  # Sky flux density, units: J s**-1 m**-3

			integrand = (
					(sky_flux_lambda/self.Egrid) * self.qe_lambda * self.unobstructed_area * self.transmission_lambda
			)  # Units e-/(m*s)
			sky_rate = trapz(integrand, self.wavgrid)  # Sky count rate at detector, units: e-/s

		else:
			sky_flux_nu = (
					(self.avg_skyBrightness*Jy)*(self.neff*self.plate_scale)**2
			)  # units of J s**-1 m**-2 Hz**-1
			
			sky_rate = (
					(sky_flux_nu/self.E0_J) * self.avg_qe * self.unobstructed_area * self.transmission * self.delta_nu
			)  # Sky count rate at detector, units: e-/s

		return sky_rate 

	def calc_src_rate(self, ab_mag):
		"""
		Calculates source count rate [units of electrons/s] given an AB magnitude. Assumes a flat SED for now.

		:param ab_mag: int, float or np.ndarray - AB magnitude of the point source
		:return: source_rate: float - Source count rate at detector, units: e-/s
		"""

		src_flux_nu = F0 * 10**(-0.4*ab_mag)  # Source flux density, units: J s**-1 m**-2 Hz**-1
		src_flux_lambda = (c / (self.lambda0 ** 2)) * src_flux_nu  # Source flux density, units: J s**-1 m**-3

		if self.integrate:
			# If magnitudes provided as an array, create a grid of flux densities to integrate over
			if isinstance(src_flux_lambda, np.ndarray):
				# Create flux density grid to vectorize integration
				grid = np.ones((ab_mag.size * self.wavgrid.size)).reshape(ab_mag.size, self.wavgrid.size)
				flux_lambda_grid = src_flux_lambda[:, np.newaxis] * grid
				integrand = (
						(flux_lambda_grid/self.E0_J) * (self.qe_lambda * self.unobstructed_area * self.transmission_lambda)
				)
				source_rate = trapz(integrand, self.wavgrid)  # Source count rate at detector, units: e-/s

			# Otherwise, single magnitude provided
			else:
				integrand = (
							(src_flux_lambda/self.E0_J) * self.qe_lambda * self.unobstructed_area * self.transmission_lambda
					)  # Units e-/(m*s)
				source_rate = trapz(integrand, self.wavgrid)  # Source count rate at detector, units: e-/s

		# If not integrating, calculate source rate using average values and bandwidth defined by delta_nu
		else:
			source_rate = (
					(src_flux_nu/self.E0_J) * self.avg_qe * self.unobstructed_area * self.transmission * self.delta_nu
			)  # Source count rate at detector, units: e-/s

		return source_rate

	def calc_snr_frame(self, ab_mag):
		"""
		Calculates SNR for a single frame given an AB magnitude for a source

		:param ab_mag: float - AB magnitude of the point source
		:return: data: dict - Dictionary containing source rate, sky rate, TSE rate, detector counts,
			and single frame SNR
		"""

		source_rate = self.calc_src_rate(ab_mag) 
		sky_rate = self.calc_sky_rate()
		detector_counts = self.calc_detector_counts()
		tse_rate = self.calc_tse_rate()

		# Calculate SNR in a single frame
		source_counts = source_rate * self.frame_time
		sky_counts = sky_rate * self.frame_time
		tse_counts = tse_rate * self.frame_time
		snr_frame = source_counts/np.sqrt(source_counts+sky_counts+tse_counts+detector_counts)

		data = dict({
			'source_rate': source_rate,
			'sky_rate': sky_rate,
			'tse_rate': tse_rate,
			'detector_counts': detector_counts,
			'snr_frame': snr_frame
		})

		return data
	# Need to allow for multiple plots
	def calc_snr(self, ab_mag, coadds, plot=False):
		"""
		Calculates SNR for the total integration given an AB magnitude for a source and a certain number of coadds

		:param ab_mag: float - AB magnitude of the point source
		:param coadds: int - Number of coadds
		:return: data: dict - Dictionary containing source rate, sky rate, TSE rate, detector counts, single frame SNR,
			and total SNR
		"""

		# Get data for a single frame
		data = self.calc_snr_frame(ab_mag)

		# Calculate SNR for the entire exposure, assumes individual frames are independent
		snr_total = data['snr_frame']*np.sqrt(coadds)

		# Add total SNR to the dictionary
		data['snr_total'] = snr_total

		if plot:
			fig, ax1 = plt.subplots()
			plt.title('{:.1f}s exposure'.format(coadds*self.frame_time))
			ax1.plot(ab_mag, snr_total)
			ax1.set_xlabel('K-dark AB Magnitude')
			ax1.set_ylabel('SNR')
			ax1.tick_params(axis='y')
			ax1.set_yscale('log')

			return data

		return data

	def calc_time(self, mag_range, snr_desired, plot=False):
		"""
		Solve for total integration time needed to reach a given SNR

		:param mag_range: np.ndarray - Array of magnitudes to plot SNR for
		:param snr_desired: float - Desired SNR
		:return: times: np.ndarray - Array of required exposure times
		"""
		data_frame = self.calc_snr_frame(mag_range)

		min_coadds = (snr_desired**2) / (data_frame['snr_frame']**2)
		times = min_coadds * self.frame_time

		if plot:
			fig, ax1 = plt.subplots()
			ax1.plot(mag_range, times)
			ax1.set_xlabel('K-dark AB Magnitude')
			ax1.set_ylabel('Required Integration Time (s)')
			ax1.tick_params(axis='y')
			ax1.set_yscale('log')
			return times

		return times
