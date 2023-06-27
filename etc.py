import numpy as np
import matplotlib.pyplot as plt
from phot_params import PhotoParameters

class PlanExposure:

	def __init__(self,paramfile):
		# Calculate effective number of pixels
		self.neff = 1#self.calc_neff()
		self.params = PhotoParameters(paramfile)


	def calc_neff(self):
		'''
		To be modified 


		LSST calculation: 
		"""
		Calculate the effective number of pixels in a single gaussian PSF.
		This equation comes from LSE-40, equation 27.
		https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

		Parameters
		----------
		fwhm_eff: `float`
		    The width of a single-gaussian that produces correct Neff for typical PSF profile.
		platescale: `float`
		    The platescale in arcseconds per pixel (0.2 for LSST)

		Returns
		-------
		nEff : `float`
		    The effective number of pixels contained in the PSF

		The fwhm_eff is a way to represent the equivalent seeing value, if the
		atmosphere could be simply represented as a single gaussian (instead of a more
		complicated von Karman profile for the atmosphere, convolved properly with the
		telescope hardware additional blurring of 0.4").
		A translation from the geometric FWHM to the fwhm_eff is provided in fwhm_geom2_fwhm_eff.
		"""
		return 2.266 * (fwhm_eff / platescale) ** 2



		For now, assume 1 pixel per target as a result of undersampling

		'''
		avg_num_pixels = 1 

		return avg_num_pixels




	def calc_instr_noise_sq(self):
	    """
	    Combine all of the noise due to intrumentation into one value

	    Parameters
	    ----------
	    params : `PhotometricParameters`
	        A PhotometricParameters object that carries details about the
	        photometric response of the telescope.

	    Returns
	    -------
	    inst_noise_sq : `float`
	        The noise due to all of these sources added in quadrature in ADU counts
	    """
	    # instrumental squared noise in units of electrons per pixel
	    inst_noise_sq =  self.neff* self.params.coadds * (self.params.read_noise**2) + (self.params.dark_current * self.params.exptime)

	    return inst_noise_sq


	def calc_sky_noise(self):
	    """
	    Calculate the noise due to sky background

	    Parameters
	    ----------
	    from LSST, to be modified --------
	    sky_sed : `Sed`
	        A Sed object representing the sky (normalized so that sky_sed.calc_mag() gives the sky brightness
	        in magnitudes per square arcsecond)
	    hardwarebandpass : `Bandpass`
	        A Bandpass object containing just the instrumentation throughputs (no atmosphere)
	    params : `PhotometricParameters`
	        A PhotometricParameters object containing information about the photometric
	        properties of the telescope.

	    Returns
	    -------
	    sky_noise : `float`
	        total non-source noise squared (in ADU counts)
	        (this is simga^2_tot * neff in equation 41 of the SNR document
	        https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40 )
	    """

	    # Sky brightness in units of ph/s/arcsec^2 at the top of the atmosphere
	    sky_brightness_phsarcsec2 = self.params.sky_brightness*self.params.Jy_toIBPhotonFlux

	 	# Sky brightness in units of ph/s/arcsec^2 at detector
	    transmitted_sky = sky_brightness_phsarcsec2*self.params.transmission
	    #print('ph/s/arcsec^2',transmitted_sky)

		#Sky background in units of electrons/s/pixel
	    sky_rate_per_sqpixel = transmitted_sky * self.params.qe * (self.neff*self.params.plate_scale)**2

	    return sky_rate_per_sqpixel

	    # # Sky background counts
	    # sky_counts = sky_rate_per_sqpixel * self.params.exptime

	    # #print('K_dark sky per pixel', sky_counts/params.exptime)
	    # self.sky_rate_per_sqpixel = sky_rate_per_sqpixel

	    # return sky_counts


	def calc_total_noise(self):
		"""
		Calculate the noise due to things that are not the source being observed
		(i.e. intrumentation and sky background)

		Parameters
		----------


		Returns
		-------
		sky_noise : `float`
		    total non-source noise squared (in ADU counts)
		    (this is simga^2_tot * neff in equation 41 of the SNR document
		    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40 )
		"""
		sky_noise = self.neff*self.calc_sky_noise() * self.params.exptime
		inst_noise_sq = self.calc_instr_noise_sq()

		total_noise_sq = sky_noise + inst_noise_sq

		return total_noise_sq


	def calc_src_counts(self,source_magnitude):

		# Brightness of Vega in units of ph/s/cm^2 at the top of the atmosphere
		# This defines the zero-point
		zp_brightness_phsarcsec2 = self.params.vega_0mag * self.params.Jy_toIBPhotonFlux

		# Vega brightness in units of ph/s at detector
		transmitted_zp = zp_brightness_phsarcsec2 * self.params.transmission 

		# Vega brightness in units of electrons/s/cm^2
		zp_ondetector = transmitted_zp * self.params.qe

		# Source count rate given source AB magnitude
		source_count_rate = zp_ondetector/(10**(source_magnitude/2.512))

		# Total source counts
		source_counts = source_count_rate*self.params.exptime

		return source_counts


	def calc_snr(self,source_magnitude):
		source_counts = self.calc_src_counts(source_magnitude)
		total_noise_sq = self.calc_total_noise()

		SNR = source_counts/np.sqrt(source_counts+total_noise_sq)

		#print('total_noise', np.sqrt(total_noise_sq))
		#print('Photons from object: ', source_counts/self.params.qe/self.params.exptime)
		#print('Photons from sky: ', self.sky_rate_per_sqpixel/self.params.qe)

		return source_counts, SNR


	def plot_snr(self,mag_range):
		snrlist = []
		srcrate_list = []

		for m in mag_range:
			srccounts, snr = self.calc_snr(m)
			snrlist.append(snr)
			srcrate_list.append(srccounts/self.params.exptime)

		snrs= np.array(snrlist)
		srcrates= np.array(srcrate_list)

		fig, ax1 = plt.subplots()
		ax1.plot(mag_range, snrs)
		ax1.set_xlabel('Magnitude')
		ax1.set_ylabel('SNR')
		ax1.tick_params(axis='y')

		return mag_range, srcrates, snrs

'''
	def plot_exptime(self,mag_range):

		something
		something
		something
'''


# for i, coadd in enumerate(coadds):
#     print('Integration Time: ',s)
#     print(snr(int_times[i],coadd))
#     print('=======')






