import numpy as np
import matplotlib.pyplot as plt
from etc import PlanExposure
from scipy.interpolate import CubicSpline

def optimize_filter(parameters,centers,bandwidths, mag):
    snrgrid = np.zeros(centers.size*bandwidths.size).reshape(centers.size,bandwidths.size)
    for i,c in enumerate(centers):
        parameters[16] = c
        for j,b in enumerate(bandwidths):
            minim = c-b/2
            maxim = c+b/2
            if (minim >= ashley_wav0[0]*1e-6) and (minim >= ashley_wav1[0]*1e-6) and (maxim <= ashley_wav0[-1]*1e-6) and (maxim <= ashley_wav1[-1]*1e-6):
                parameters[17] = b
                plan = PlanExposure(param_arr=parameters)
                data= plan.calc_snr(AB_magnitude=mag)
                snrgrid[i][j] = data[-1]
            else:
                snrgrid[i][j] = np.nan
    return snrgrid


def create_snrgrid(parameters,loweredge,higheredge, mag):
    snrgrid = np.zeros(loweredge.size*higheredge.size).reshape(loweredge.size,higheredge.size)
    for i,hi in enumerate(higheredge):
        for j,low in enumerate(loweredge):
            if hi > low:
                c = (hi-low)/2 + low
                b = hi-low
                parameters[16] = c
                minim = c-b/2
                maxim = c+b/2
                if (minim >= ashley_wav0[0]*1e-6) and (minim >= ashley_wav1[0]*1e-6) and (maxim <= ashley_wav0[-1]*1e-6) and (maxim <= ashley_wav1[-1]*1e-6):
                    parameters[17] = b
                    plan = PlanExposure(param_arr=parameters)
                    data= plan.calc_snr(AB_magnitude=mag)
                    snrgrid[i][j] = data[-1]
                else:
                    snrgrid[i][j] = np.nan
            else:
                snrgrid[i][j] = np.nan

    return snrgrid


def optimize_filter(snrgrid, higheredge, loweredge, xlo, xhi, ylo, yhi, x_tick_positions, y_tick_positions, plot=True):
    besthiid, bestloid = np.where(snrgrid == np.nanmax(snrgrid))
    besthi = higheredge[besthiid[0]] / 1e-6
    bestlo = loweredge[bestloid[0]] / 1e-6
    bestcenter = (besthi - bestlo) / 2 + bestlo
    bestbw = besthi - bestlo

    ids = np.array([bestloid[0], besthiid[0]])
    lam_minmax = np.array([bestlo, besthi])
    cen_bw = np.array([bestcenter, bestbw])

    print('Optimum filter edges: {:.3f}, {:.3f}'.format(bestlo, besthi))
    print('Optimum filter center,bw: {:.3f}, {:.3f}'.format(bestcenter, bestbw))

    if plot:
        plt.imshow(snrgrid, vmin=0.9 * np.nanmax(snrgrid), vmax=np.nanmax(snrgrid))
        cbar = plt.colorbar()
        cbar.set_label("SNR")
        plt.scatter(bestloid, besthiid, color='r')

        plt.xlim(xlo, xhi)
        plt.ylim(ylo, yhi)

        # # Set custom ticks and labels for x and y axes
        x_tick_labels = np.round(loweredge[x_tick_positions] * 1e6, 2)
        y_tick_labels = np.round(loweredge[y_tick_positions] * 1e6, 2)

        plt.xticks(x_tick_positions, x_tick_labels)
        plt.yticks(y_tick_positions, y_tick_labels)

        plt.axvline(bestloid, color='r')
        plt.axhline(besthiid, color='r')

        plt.ylabel('$\lambda_{max}$  ($\mu$m)')
        plt.xlabel('$\lambda_{min}$  ($\mu$m)')

    return ids, lam_minmax, cen_bw