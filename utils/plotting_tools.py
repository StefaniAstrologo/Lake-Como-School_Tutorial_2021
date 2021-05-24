
################################### Color functions
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from utils.brownian_function import brownian
from scipy.integrate import odeint

def Stoch_Plot(time_steps, mRNA_trajectory):

    average_line = len(mRNA_trajectory)*[np.mean(mRNA_trajectory)]
    plt.rc('font', size = (14))
    plt.figure(figsize=(12,5))
    hfont = {'fontname':'Georgia'}
    plt.plot(time_steps, mRNA_trajectory, 'b-', label='mRNA')
    plt.plot(time_steps, average_line, 'k--', label='average')
    plt.xlabel('Time') , plt.ylabel('# mRNA')
    plt.legend(loc='upper right', shadow=True)
#     plt.xlim(0.0,50)
    plt.show()



def ODE_Plot(k_burst,k_deg,burst_size):
    params = [k_burst,k_deg, burst_size]
    # Initial conditions
    mRNA_0 = 0
    state = (mRNA_0)

    # Time
    dt = 0.1 # step size!
    t = np.arange(0,50,dt)

    # We define a function ODE system (in format used for odeint)
    def ODE(state,t,params):

        kb, kd, b_size = params

        mRNA           = state

        # rates
        vt             = kb*b_size - kd*mRNA

        # equations
        dmRNA_dt       = vt

        return (dmRNA_dt)

    # Solve using odeint
    solution = odeint(ODE,state,t,args=(params,)) 
    tmRNA = solution[:,0]

    # Show over time
    plt.rc('font', size = (14))
    plt.figure(figsize=(12,5))
    plt.plot(t, tmRNA, 'b-', label='mRNA')
    plt.xlabel('Time') , plt.ylabel('# mRNA')
    plt.legend(loc='upper right', shadow=True)
    plt.show()
    


def hist_bins(data):
    """Create a list of """
    bins = np.arange(0, max(data) + 1) - 0.5 
    return(bins)

def mod_array_for_gradient_color(arr, segment_size=10):
    """
    This function split each part of the array in segments so it can be assign a different color to each segment
    each segment is a subarray
    """
    sublist_overlapping = lambda list_, chunk_size, overlap : [list_[i:i+chunk_size] for i in range(0, len(list_), chunk_size-overlap)]
    result = []
    result.append(sublist_overlapping(arr[0], chunk_size=segment_size, overlap=1))
    result.append(sublist_overlapping(arr[1], chunk_size=segment_size, overlap=1))
    return result

def plot_gradient_line(x, ax, segment_size=23, colorLimits=['green', 'red'], label='trajectory'):
    """
    It takes the original x array and split it in segments of specific size (here there is a low limit that has to be
    explore by hand, not sure how to check this automatically). For you this arrays is 23. If you increase this number, the amount
    of colors for the gradient gets reduced. So the maximum number of colors here is for 23 segment size.
    ColorLimit stablish the which colors to use for transition. 
    Label the name of the line ploted.
    """
    from colour import Color   # This tool is nice for gradient line --> pip install colour
    xMod = mod_array_for_gradient_color(x, segment_size=segment_size) # This line modify the array to be plot in gradient
    red = Color(colorLimits[0])
    colors = list(red.range_to(Color(colorLimits[1]),segment_size))
    colorsHex = [i.get_hex() for i in colors]

    for ix, i in enumerate(range(len(xMod[0]))):
        p1, = ax.plot(xMod[0][i], xMod[1][i], color=colorsHex[ix], label=label)
    
    p2, = ax.plot(x[0,0],x[1,0], 'g^', label = 'starting point', markersize=10, color='blue')
    p3, = ax.plot(x[0,-1], x[1,-1], 'r^', label = 'end point', markersize=10, color='black')
    return [p1, p2, p3]

def plot_loop_subplot(x, b_params, in_color=False, savein=None):
    """
    Plot the 2D trajectory of brownian motion
    x: Array
    in_color: False if in black, True if color gradient (requires pip install colour)
    """
    hfont = {'size':20}
    title = '2D Brownian Motion'
    n = ceil(np.sqrt(b_params['n_loops']))
    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(25,25))
    axs = axs.ravel()
    plt.suptitle("{}".format(title), **hfont, y=0.92)
    plt.subplots_adjust(hspace=0.7, wspace=0.7)

    for ix, i in enumerate(range(b_params['n_loops'])):
            brownian(x[:,0], b_params['N'], b_params['dt'], b_params['delta'], out=x[:,1:])

            # Color gradient
            if in_color == True:
                p1, p2, p3 = plot_gradient_line(x=x, ax=axs[ix],
                            segment_size=23, colorLimits=['green', 'red'], label='trajectory',)

                axs[ix].set_xticks([]), axs[ix].set_yticks([])
            else:
                
                # Your usual method
                p1, = axs[ix].plot(x[0],x[1], 'k',label = 'tarjectory')
                axs[ix].set_xticks([]), axs[ix].set_yticks([])
                # Mark the start and end points.
                p2, = axs[ix].plot(x[0,0],x[1,0], 'g^', label = 'starting point')
                p3, = axs[ix].plot(x[0,-1], x[1,-1], 'r^', label = 'end point')

            # Other plot specifications
            #     axs[ix].rc('font', size = (10))
            axs[ix].set_title('Simulation {}'.format(ix+1),**hfont)
            axs[ix].set_xlabel('x', **hfont), axs[ix].set_ylabel('y', **hfont)
            axs[ix].axis('equal')
            axs[ix].grid(True)
            axs[ix].legend(handles=[p1, p2, p3], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    for ix in np.arange(n**2)[::-1][:-b_params['n_loops']]: # Removign axis without the plot
        axs[ix].axis('off')
        
    if savein != None:
        plt.savefig(savein, dpi=300, bbox_inches = 'tight')