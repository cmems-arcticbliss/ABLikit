"""My SEA ICE tools
A collection of  i often use when analysing sea ice model outputs (NEMO based). 
"""


## standart libraries

import os,sys
import numpy as np
from scipy import stats

# xarray
import xarray as xr

# plot
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

from matplotlib import cm 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

import cartopy.feature as cfeature
import copy
import cmocean

print(f"Name: {__name__}")
print(f"Package: {__package__}")

print("This is a collection of  tools i often use when analysing sea ice NEMO based outputs.")


#def __init__():
#    print('Init')

def main():
    print('Create example map (global) without any data.')
    #set default grid parameters before plotting
    gridparam = Fpltgridparamdefaults(reg='GLO')
    print('- gridparam')
    print(gridparam)
    # plot
    outo = FpltGLO(0,pltgridparam=gridparam,ty='DIFF',saveo=True,pltshow=False)


def Fload_experiments(experiments, diribase, prefix="NANUK4_ICE_ABL-", freq="1h", dirigrid=None):
    """
    Load datasets for a list of experiment configurations.

    Parameters:
    experiments (list of dict): List of experiment configurations. Each dictionary should contain:
        - 'frc' (str): Forcing type (e.g., "ABL" or "BLK").
        - 'rheol' (str): Rheology type (e.g., "EVP" or "BBM").
        - 'nb' (str): Experiment number (e.g., "903" or "904").
        - 'loadice' (bool, optional): Whether to load ice data. Default is True.
        - 'loadoce' (bool, optional): Whether to load ocean data. Default is False.
        - 'loadall' (bool, optional): Whether to load all data. Default is False.
    diribase (str): Base directory for the datasets.
    prefix (str): Prefix for the dataset files. Default is "NANUK4_ICE_ABL-".
    freq (str): Frequency of the data. Default is "1h".
    dirigrid (str): Directory for the grid files.

    Returns:
    dict: Dictionary where keys are experiment names and values are the loaded datasets.

    Examples:
    --------
    experiments = [
        {"frc": "ABL", "rheol": "EVP", "nb": "903", "loadice": True, "loadoce": False, "loadall": True},
        {"frc": "BLK", "rheol": "EVP", "nb": "903", "loadice": True, "loadoce": True, "loadall": False},
        {"frc": "ABL", "rheol": "BBM", "nb": "903", "loadice": True, "loadoce": False, "loadall": True},
        {"frc": "BLK", "rheol": "BBM", "nb": "903", "loadice": True, "loadoce": True, "loadall": False},
        {"frc": "ABL", "rheol": "BBM", "nb": "904", "loadice": True, "loadoce": False, "loadall": True},
        {"frc": "BLK", "rheol": "BBM", "nb": "904", "loadice": True, "loadoce": True, "loadall": False}
    ]
    diribase = "/gpfsstore/rech/cli/regi915/NEMO/NANUK4/"

    # Calling the function
    experiment_datasets = load_experiments(experiments, diribase)

    # Using the Dictionnary of datasets:
    experiment_datasets["ABLEVP903"].datice
    """
    
    if dirigrid is None:
        dirigrid = "/gpfsstore/rech/cli/regi915/NEMO/NANUK4/NANUK4.L31-I/"
    
    experiment_datasets = {}

    # Load all experiments
    for exp in experiments:
        exp_name = f"{exp['frc']}{exp['rheol']}{exp['nb']}"
        #print(f"Loading experiment: {exp_name}")
        experiment_datasets[exp_name] = expe(
            diribase,
            exp['frc'],
            exp['rheol'],
            exp['nb'],
            loadice=exp.get('loadice', True),
            loadoce=exp.get('loadoce', False),
            loadall=exp.get('loadall', False),
            prefix=prefix,
            freq=freq,
            dirigrid=dirigrid
        )
    
    return experiment_datasets

def Faddhistodat(dat,histo=True,nbins=[0,0.5,1],histtype='stepfilled',li=1,co="b",label="",al=1,pval=0.95,com='k',cop='k',lim='3',lip='2',alm=0.5,alp=0.7,listm='-',listp='-'):
    if histo:
        dat.plot.hist(bins=nbins,density=True,histtype=histtype,linewidth=li,alpha=al,color=co,label=label)
    meanval,stdval,p95val,skewval = Fstats(dat,pval=pval)
    plt.axvline(meanval, color=com, linestyle=listm, linewidth=lim,alpha=alm,zorder=10)
    plt.axvline(p95val, color=cop, linestyle=listp, linewidth=lip,alpha=alp,zorder=10)
    return meanval,p95val


def FstatsOLD(seldat,pval=0.95):
    STack=seldat.stack(z=("x", "y","time_counter"))
    meanval = STack.mean(dim="z")
    p95val  = STack.quantile(q=pval,skipna=True,dim="z")
    stdval  = STack.std(dim="z")
    return meanval,stdval,p95val

def Fstats(seldat,pval=95):
    # Step 1: Convert to a NumPy array
    data_np = seldat.values
    # Step 2: Flatten the NumPy array
    data_flat = data_np.flatten()
    # Step 3: Compute the stats
    skewval = float(stats.skew(data_flat,nan_policy='omit').data)
    meanval = data_flat[~np.isnan(data_flat)].mean()
    p95val  = np.nanpercentile(data_flat,q=95)
    stdval  = data_flat[~np.isnan(data_flat)].std()
    return meanval,stdval,p95val,skewval



def FcomputeGeoM(data,e1,e2):
        """ Compute the geographic mean of a field at each time step.
            
            Parameters:
            - data (float): xarray of dimension x,y,t 
            - e1 (float): xarray of dimension x,y that is the size of the data mesh in x direction
            - e2 (float): xarray of dimension x,y that is the size of the data mesh in y direction
    
            Return: gm (float) xarray of dimention t containing the global mean value
        """
        
        weighted_data =  data * e1.squeeze() 
        weighted_data =  weighted_data * e2.squeeze() 
        stacked_weighted_data = weighted_data.stack(z=("x", "y"))
        weights = e1 * e2
        weights = weights.squeeze()
        stacked_weights = weights.stack(z=("x", "y"))
        summed_weights = stacked_weights.sum(dim='z')
        gm = stacked_weighted_data.sum(dim="z") / summed_weights
        
        return gm
    
def Fselectreg(dat,meshmask,rect1):
    """ select subregion from data array and metrics
    """
    if (len(rect1)==4):
        rectdat = dat.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
        recte1 = meshmask.e1t.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
        recte2 = meshmask.e2t.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
    return rectdat,recte1,recte2

#def Fselectdistmask(dat,meshmask,distmask):
#    """ select subregion from data array and metrics
#    """
#    seldat = dat.where(distmask.tmask>=1).load()
#    sele1 = meshmask.e1t.where(distmask.tmask>=1).load()
#    sele2 = meshmask.e2t.where(distmask.tmask>=1).load()
#    #seldat.assign_attrs(expe=dat.namexp)
#    return seldat,sele1,sele2

def Fselectdistmask(datarr,typ,var2sp,distmask):
    """ select subregion from data array and metrics
    """
    if typ=='datoce':
        dat = datarr.datoce[var2sp]
        varnam=datarr.datoce[var2sp].long_name
        varunits=datarr.datoce[var2sp].units
    if typ=='datice':
        dat = datarr.datice[var2sp]
        varnam=datarr.datice[var2sp].long_name
        varunits=datarr.datice[var2sp].units
    if typ=='databl':
        dat = datarr.databl[var2sp]
        varnam=datarr.databl[var2sp].long_name
        varunits=datarr.databl[var2sp].units
    meshmask = datarr.meshmask
    
    seldat = dat.where(distmask.tmask>=1).load()
    sele1 = meshmask.e1t.where(distmask.tmask>=1).load()
    sele2 = meshmask.e2t.where(distmask.tmask>=1).load()
    # 
    seldat.attrs['expe'] = datarr.namexp
    seldat.attrs['Cd'] = datarr.Cd
    seldat.attrs['varnam'] = varnam
    seldat.attrs['varunits'] = varunits
    
    return seldat,sele1,sele2


def Fcomputetsm(dat,meshmask,rect1):
    """ Compute mean time series from data array and metrics
    """
    if (len(rect1)==4):
        rectdat = dat.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
        recte1 = meshmask.e1t.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
        recte2 = meshmask.e2t.isel(x=slice(rect1[0],rect1[1]+1),y=slice(rect1[2],rect1[3]+1)).load()
        tsm = FcomputeGeoM(rectdat,recte1,recte2)
    elif (len(rect1)==2):
        tsm = dat.isel(x=rect1[0],y=rect1[1]).load()
    else:
        print("ERROR, rect1 should provide 4 coordinates [x1,x2,y1,y2] or 2 coordinates [x,y]")
        

    return tsm
    
def Fpltcolorbar(fig1,ax,var2plt,norm,cmap,cblev,tlabel,textco='w',F4P=False,F2P=False,extend="no"):

        if (extend =="no"):    # else defined in parameters
            if var2plt=='siconc':
                extend='neither'
            elif var2plt=='qt_ice':
                extend='both'
            else:
                extend='max'

        if (F4P==True):
            axins1 = inset_axes(ax,
                        height="70%",  # height : 5%
                        width="10%",
                        bbox_to_anchor=(0.89, -0.1,0.2,0.9),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
            cb = fig1.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),cax=axins1, orientation='vertical',ticks=cblev,label=tlabel,extend=extend)  
            sizlab=10
            sizti=10
            col=textco
            
        else:
            axins1 = inset_axes(ax,
                        height="10%",  # height : 5%
                        width="50%",
                        bbox_to_anchor=(0.08, 0.79,0.9,0.2),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
            cb = fig1.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),cax=axins1, orientation='horizontal',ticks=cblev,label=tlabel,extend=extend) 
            sizlab=14
            sizti=12
            col=textco
            
                
        # set colorbar label plus label color
        cb.set_label(label=tlabel,color=col,size=sizlab)

        # set colorbar tick color
        cb.ax.xaxis.set_tick_params(color=col,size=sizti)

        # set colorbar edgecolor 
        cb.outline.set_edgecolor(col)

        # set colorbar ticklabels
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=col)

        return cb
    
def Fsetcmapnorm(var2plt,vmin=-1,vmax=1,cblev=[0],linnorm=True):
            
        # color map field 
        if ((var2plt=='siconc' or var2plt=='a_i') or var2plt=='a_ip'):
            if len(cblev)==1:
                cblev=[0,0.6,0.8,0.9,0.95,0.99]
            cmap = copy.copy(cmocean.cm.ice)
            cmap.set_bad('r',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.PowerNorm(gamma=4.5,vmin=vmin, vmax=vmax)

            if linnorm: 
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
        # color map field 
        elif var2plt=='windsp':
            if len(cblev)==1:
                cblev=[0,0.7,0.8,0.85,0.9,0.95,0.99]
            cmap = cm.Spectral_rcmocean.cm.ice
            cmap.set_bad('k',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
 
        elif var2plt=='qt_ice':
            if len(cblev)==1:
                cblev=[-200,-100,0]
            cmap = cmocean.cm.matter
            cmap.set_bad('k',1.)
            cmap.set_under('w')
            cmap.set_over('k')
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # color map field 
        elif ((var2plt=='taum')|(var2plt=='taum_ice')):
            if len(cblev)==1:
                cblev=[0,0.7,0.8,0.85,0.9,0.95,0.99]
            cmap = cm.Spectral_r
            cmap.set_bad('k',1.)
            cmap.set_under('w')
            cmap.set_over('k')
            #norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            norm = mcolors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax)
            

        # color map field 
        elif (var2plt=='sivelo-t'):
            if len(cblev)==1:
                print(cblev)
                cblev=[0,0.1,0.2]
            print(cblev)
            #cmap = cm.RdYlGn_r
            cmap = cmocean.cm.thermal
            cmap.set_bad('k',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
        # color map field 
        elif var2plt=='sidefo-t':
            if len(cblev)==1:
                cblev=[0,0.001e-4,0.01e-4,0.025e-4,0.05e-4,0.1e-4,0.2e-4,0.3e-4]
            cmap = cmocean.cm.thermal
            cmap.set_bad('r',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax)
            
        elif (var2plt=='pblh'):
            if len(cblev)==1:
                cblev=[0,0.7,0.8,0.85,0.9,0.95,0.99]
            cmap = cm.Spectral_r
            cmap.set_bad('k',1.)
            cmap.set_under('w')
            cmap.set_over('k')
            #norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            norm = mcolors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax)
            
        else :
            if len(cblev)==1:
                cblev=[0,0.7,0.8,0.85,0.9,0.95,0.99]
            cmap = copy.copy(cm.get_cmap("Spectral"))
            cmap.set_bad('k',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm,cblev
    
def Fplot4panels(d1,d2,d3,d4,var2plt,it,pltshow=True,logo=True,pltsave=True,diro='./',ty='T',varty=1,maskoce=False,cmapforced="no",alphalogo=0.3,dpifig=300,cbar=True,sicol='no',sicolwdth=1,cblev=[0],pltzoom=False,x1=0,x2=0,x3=0,x4=0,rectcol='r',vmax=0.02,vmin=0,Lzoom=False,zoom=[0,0,0,0],textco='k',pltzone=False,distmask=0,threshmask=0.15):
        """
        Plot 4-panel figures with 4 maps.
    
        Parameters:
        ----------
        d1, d2, d3, d4: xarray.Dataset
            Datasets containing the data (experiments) to plot.
        var2plt : str
            Variable to plot.
        it : int
            Time index at which the maps should be plotted
        pltshow : bool, optional
            Whether to display the plot. Default is True.
        logo : bool, optional
            Whether to add a Datlas logo. Default is True.
        alphalogo : float, optional
            Alpha value for the logo. Default is 0.3.
        pltsave : bool, optional
            Whether to save the plot. Default is True.
        diro : str, optional
            Directory to save the plot. Default is './'.
        dpifig : int, optional
            Resolution (dpi) of the saved plot. Default is 300.
        ty : str, optional
            Type of data grid ('T', 'U', 'V', 'F'). Default is 'T'.
        varty : int, optional
            Variable type (1: ice, 2: ocean, 3: ABL). Default is 1.
        maskoce : bool, optional
            Whether to mask ocean data. Default is False.
    
        Lzoom : bool, optional
            Whether to use predefined zoom settings instead of full area. Default is False.
        zoom : list, optional
            Zoom coordinates [x_min, x_max, y_min, y_max]. Default is [0, 0, 0, 0].
            
        cbar : bool, optional
            Whether to add a colorbar. Default is True.
        cblev : list, optional
            Colorbar levels. Default is [0].
        vmax : float, optional
            Maximum value for the color scale. Default is 0.02.
        vmin : float, optional
            Minimum value for the color scale. Default is 0.
    
        sicol : str, optional
            Color for sea ice contours. Default is 'r'.
        sicolwdth : int, optional
            Line width for sea ice contours. Default is 1.
            
        pltzoom : bool, optional
            Whether to plot a rectangle on map to represent a rectangular area. Default is False.
        x1, x2, x3, x4 : int, optional
            Coordinates for rectangle area. Defaults are 0.
        rectcol : str, optional
            Color for zoom rectangle. Default is 'r'.
            
        pltzone : bool, optional
            Whether to plot line to represent a specific area on the map. Default is False.
        distmask : int, optional
            Distance mask value. Default is 0.
    
        textco : str, optional
            Color for text annotations. Default is 'k'.
    
        Returns:
        -------
        fig1 : matplotlib.figure.Figure
            The figure object.
        ax : numpy.ndarray
            Array of axis objects.
        cs : list
            List of contour plot objects.
        cs2 : list
            List of sea ice contour plot objects.
        cb : list
            List of colorbar objects.
    
        Examples:
        --------
        * 4 panels over full arctic, and dotted line for the dist2coast subregion:
        >>> allexps = li.Fload_experiments(experiments, diribase)
        >>> fig1,ax = li.Fplot4panels(allexps['ABLEVP903'],allexps['BLKEVP903'],allexps['ABLBBM903'],allexps['BLKBBM903'],'windsp',it,pltshow=True,diro=diro+"maps/",pltsave=False,varty=2,sicol='r',vmin=vmax,vmax=vmin,cblev=cblev,dpifig=600,pltzone=True,distmask=distmask,Lzoom=False,zoom=rect1,textco='k')
        * 4 panels over a sburegions centered on the dist2coast area:
"""
        
        indices = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
        # get mask
        if ty=='U':
            mask = d1.meshmask.umask[0,0,:,:]
        if ty=='V':
            mask = d1.meshmask.vmask[0,0,:,:]
        if ty=='F':
            mask = d1.meshmask.fmask[0,0,:,:]
        else:
            mask = d1.meshmask.tmask[0,0,:,:]
            
        # Get ice contour from sea ice concentration
        varsice = 'siconc'
        datasets = [d1, d2, d3, d4]
        icedat = [ds.datice[varsice].isel(time_counter=it).where(mask != 0) for ds in datasets]
        NT = d1.datice.time_counter.size
        tdate = d1.datice.time_counter.to_index()[it]
    
    
        # Mask ocean if required
        if maskoce:
            masks = [ds.datice['siconc'].isel(time_counter=it).where(mask != 0) for ds in datasets]

        # Get data to plot based on varty
        if varty == 1:
            data2plot = [ds.datice[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
            tlabel = d1.datice[var2plt].long_name + " (" + d1.datice[var2plt].units + ")"
        elif varty == 2:
            data2plot = [ds.datoce[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
            tlabel = d1.datoce[var2plt].long_name + " (" + d1.datoce[var2plt].units + ")"
        elif varty == 3:
            data2plot = [ds.databl[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
            tlabel = d1.databl[var2plt].long_name + " (" + d1.databl[var2plt].units + ")"

        # Apply mask if required
        if maskoce:
            data2plot = [data.where(mask > 0.15) for data, mask in zip(data2plot, masks)]
        
        if Lzoom:
            gridinc=50
            # name of plot to output
            namo="4panZOOM_"+d1.prefix+d1.freq+"_"+var2plt+"_"+str(it).zfill(4)    
        else:
            gridinc=100
            # name of plot to output
            namo="4pan_"+d1.prefix+d1.freq+"_"+var2plt+"_"+str(it).zfill(4)    


        # plot title
        titles = [
            d1.frc[0:3] + "+" + d1.rheol,
            d2.frc[0:3] + "+" + d2.rheol,
            d3.frc[0:3] + "+" + d3.rheol,
            d4.frc[0:3] + "+" + d4.rheol
            ]
        
        cmap,norm,cblev = Fsetcmapnorm(var2plt,vmin=vmin,vmax=vmax,cblev=cblev)
        if (cmapforced != "no"):
            cmap = copy.copy(cm.get_cmap(cmapforced))

        # Main plotopen_dataset
        fig1, ax = plt.subplots(2, 2, figsize=[12, 12], facecolor='w', gridspec_kw={'width_ratios': [1, 1]})

        for (i, j), data, ice in zip(indices, data2plot, icedat):
                cs, ax[i, j] = FplotmapSI_gp(fig1, ax[i, j], data, cmap, norm, plto='tmp_plot', gridpts=True, gridptsgrid=True, gridinc=gridinc, gstyle='darkstyle')
                if sicol != "no":
                    cs_ice = ax[i, j].contour(ice, alpha=0.9, colors=sicol, linestyles="-", linewidths=sicolwdth, levels=np.arange(threshmask,threshmask+0.01,threshmask)) 
                else:
                    cs2 = None
                    
        if Lzoom:
            #--- set axes for each subplot
            for i in range(2):
                for j in range(2):
                    ax[i,j].set_xlim([zoom[0],zoom[1]])
                    ax[i,j].set_ylim([zoom[2],zoom[3]])
                    
            #--- display date
            tcolordate=textco 
            tsizedate=10
            for i in range(2):
                for j in range(2):
                    ax[i,j].annotate(tdate,xy=(250,530),xycoords='data', color=tcolordate,size=tsizedate)
                    
            #--- display title
            ticol=textco
            tisi=12
            for (i, j), title in zip(indices, titles):
                ax[i, j].annotate(title, xy=(120,530), xycoords='data', color=ticol, size=tisi)
        else: # if all arctic
             #--- display date
            tcolordate="#dfdfdf"
            tsizedate=10
            for i in range(2):
                for j in range(2):
                    ax[i,j].annotate(tdate,xy=(20,520),xycoords='data', color=tcolordate,size=tsizedate)
            #--- display title
            ticol='w'
            tisi=12
            for (i, j), title in zip(indices, titles):
                ax[i, j].annotate(title, xy=(380, 543), xycoords='data', color=ticol, size=tisi)

        # add colorbar
        if cbar:
            if Lzoom:
                barco='k'
            else:
                barco=textco
                
            cb3 = Fpltcolorbar(fig1,ax[0,1],var2plt,norm,cmap,cblev,tlabel,F4P=True,textco=barco)
            cb4 = Fpltcolorbar(fig1,ax[1,1],var2plt,norm,cmap,cblev,tlabel,F4P=True,textco=barco)
           
    
    
        # add Datlas logo
        if logo:
            FaddDatlasLogo(fig1,alpha=alphalogo,NbP=4)

        # Whether to plot a rectangle on map to represent a rectangular area. Default is False.
        if pltzoom:
            for i in range(2):
                for j in range(2):
                    rect = patches.Rectangle((x1, x2), x3, x4, linewidth=1, edgecolor=rectcol, facecolor='none', zorder=20)
                    ax[i, j].add_patch(rect)

        # Whether to plot line to represent a specific area on the map. Default is False.
        if pltzone:
            for i in range(2):
                for j in range(2):
                    cssel   = ax[i,j].contour(distmask.tmask,alpha=0.7,colors='k',linestyles="--",linewidths=1.5,levels=np.arange(1.,2,1.5))
        
    
        plt.subplots_adjust(
                    wspace=0.1, 
                    hspace=0.1)
        
        if pltshow:
            plt.show()

        if pltsave:    
        # Save fig in png, resolution dpi    
            Fsaveplt(fig1,diro,namo,dpifig=dpifig)
            plt.close(fig1)
            
        return fig1,ax


def FplotSP(freqs,spdat2plot,xmin,xmax,sp2=[0.],sp3= [0.],sp4=[0.],sp5=[0.],co='#DF3A01',co2='#0B4C5F',co3='#D0FA58',co4='k',co5='k',ti1='no',ti2='no',ti3='no',ti4='no',ti5='no',title='no'):
        """
        Plot time spectrum (up to 5 lines).
        """
        ax2 =  plt.gca() 

        l1 = plt.plot(freqs,spdat2plot,co,linewidth=2.0,label=ti1)
        print('ts1')
        if any(sp2)!=0:
            print('ts2')
            l2 = plt.plot(freqs,sp2,co2,linewidth=1.5,alpha=0.5,linestyle="--",label=ti2)
        if any(sp3)!=0:
            print('ts3')
            l3 = plt.plot(freqs,sp3,co3,linewidth=1.0,alpha=0.9,linestyle="-",label=ti3)  
        if any(sp4)!=0:
            print('ts4')
            l4 = plt.plot(freqs,sp4,co4,linewidth=0.8,alpha=0.7,linestyle=":",label=ti4)  
        if any(sp5)!=0:
            print('ts5')
            l5 = plt.plot(freqs,sp5,co5,linewidth=1,alpha=0.5,linestyle="-",label=ti5)  
            
        ax2.set_xscale('log', nonposx='clip')
        ax2.set_yscale('log', nonposy='clip')

        #---- X-Ticks 
        ax2.set_xlabel('Frequency (cph)', fontsize=12)
        ax2.xaxis.label.set_size(13)
        ax2.set_xlim(10 ** xmin, 10 ** xmax)\

        #---- Second axis with spatial wave lengths 
        twiny = ax2.twiny()

        twiny.set_xscale('log', nonposx='clip')
        twiny.set_xlim(10 ** xmin, 10 ** xmax)

        # major ticks
        new_major_ticks = 10 ** np.arange(xmin+1 , xmax, 1.)
        new_major_ticklabels = 1. / new_major_ticks
        new_major_ticklabels = ["%.0f" % i for i in new_major_ticklabels]

        twiny.set_xticks(new_major_ticks)
        twiny.set_xticklabels(new_major_ticklabels, rotation=60, fontsize=12)

        # minor ticks
        A = np.arange(2, 10, 2)[np.newaxis]
        B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
        C = np.dot(B.transpose(), A)

        new_minor_ticklabels = C.flatten()
        new_minor_ticks = 1. / new_minor_ticklabels
        new_minor_ticklabels = ["%.0f" % i for i in new_minor_ticklabels]
        twiny.set_xticks(new_minor_ticks, minor=True)
        twiny.set_xticklabels(new_minor_ticklabels, minor=True, rotation=60,
                              fontsize=12)
        twiny.set_xlabel('Period (hour)', fontsize=12)

        # Y-Axis
        ax2.set_ylabel('PSD (m$^2$ cph$^{-1}$)', fontsize=12)
        
        # Title
        if ti1!='no':
            ax2.legend(loc=3)
        
        if title!='no':
            plt.title(title)
        
        # Grid
        ax2.grid(True, which='both')

        return ax2
    
def Ffindij(dirigrid,lonval, latval):
    """Finds i, j coordinates nearest to latitude and longitude values.
    
    Parameters:
    - dirigrid (str): grid files where to read nav_lon nav_lat.
    - latval (float): latitude value requested.
    - lonval (float): longitude value requested.
    
    Returns:
    tuple: Tuple of integers (i, j).
    """
    lat = xr.open_dataset(dirigrid+'mesh_hgr.nc')['nav_lat']
    lon = xr.open_dataset(dirigrid+'mesh_hgr.nc')['nav_lon']     

    # Find the index of the grid point nearest a specific lat/lon.   
    abslat = np.abs(lat-latval)
    abslon = np.abs(lon-lonval)
    c = np.maximum(abslon, abslat)

    ([j], [i]) = np.where(c == np.min(c))
    
    return i,j
    
    
def Fstrmb(ie):
    """ Reads an integer between 1-1000 and returns a string with 3, 2 or 1 zeroes before int so that there are 4 characters in total.
            
        Parameters:
        - ie (int): integer between 1-1000
        
        Returns:
        str: name with 3 zeroes before integer.
        """
    if ie<10:
        ena = "000"+str(ie)
    elif ((ie>9)&(ie<100)):
        ena = "00"+str(ie)
    elif ((ie>99)&(ie<1000)):
        ena= "0"+str(ie)
    else:
        ena= str(ie)
    return ena


    

def Faddcolorbar(fig,cs,ax,levbounds,levincr=1,tformat="%.2f",tlabel='',shrink=0.45,facmul=1.,orientation='vertical',tc='k',loc='lower right',wth="15%",bbta=(0.08, -0.1,0.9,0.2)):
    """Function to add a customized colorbar to a figure.
    
    Parameters:
    - fig: Figure properties.
    - cs: Plotted field properties.
    - ax: Axis properties.
    - levbounds (array): [min, max, incr] min, max, and increment of the requested colormap.
    - levincr (int): Increment at which to display colorbar labels.
    - tformat (str): Default is "%.2f".
    - tlabel (str): Label to display near the colorbar.
    - shrink (float): Proportion in [0,1] for the relative size of the colorbar compared to axis.
    - facmul (float): Multiplication factor for displaying colorbar labels in a different unit.
    - orientation (str): 'horizontal' or 'vertical'.
    - tc (str): Color of text labels and ticks.
    - loc (str): Position of the colorbar (default is 'lower right').
    - wth (str): Percentage setting the height of a horizontal colorbar.
    - bbta (tuple): Bounding box defining the location of the colorbar. Default is (0.08, -0.1,0.9,0.2).
    
    Returns:
    dict: Colorbar properties (cb).
    """
    
    lmin = levbounds[0]
    lmax = levbounds[1]
    incr = levbounds[2]
    levels = np.arange(lmin,lmax,incr)
    cblev = levels[::levincr]
    
    if orientation =='horizontal':
        axins1 = inset_axes(ax,
                        height=wth,  # height : 5%
                            width="50%",
                        bbox_to_anchor=bbta,
                        bbox_transform=ax.transAxes,
                        borderpad=0)

    if orientation =='vertical':
        axins1 = inset_axes(ax,
                        height="50%",  # height : 5%
                            width="2%",
                        loc='center left',
                       borderpad=2)

    cb = fig.colorbar(cs,cax=axins1,
                                    extend='both',                   
                                    ticks=cblev,
                                    spacing='uniform',
                                    orientation=orientation,
                                    )
    
    new_tickslabels = [tformat % i for i in cblev*facmul]
    cb.set_ticklabels(new_tickslabels)
    cb.ax.set_xticklabels(new_tickslabels, rotation=70,size=10,color=tc)
    cb.ax.tick_params(labelsize=10,color=tc) 
    cb.set_label(tlabel,size=14,color=tc)
    
    return cb
    
def Fplothello():
    """Add Datlas logo to the figure.
    
    Parameters:
    - fig1: Figure properties.
    - alpha (float): Transparency of the logo.
    - path (str): Path to the Datlas logo image.
    
    Returns:
    None
    """
    # 
    print("hello")
    
def FaddDatlasLogo(fig1,alpha=0.3,path='/linkhome/rech/genige01/regi915/logo-datlas-RVB-blanc.png',NbP=1):
    """Add Datlas logo to the figure.
    
    Parameters:
    - fig1: Figure properties.
    - alpha (float): Transparency of the logo.
    - path (str): Path to the Datlas logo image.
    
    Returns:
    None
    """
    # add Datlas logo
    im = plt.imread(path)
    if NbP==1:
        newax = fig1.add_axes([0.15,0.15,0.06,0.06], anchor='SW', zorder=10)
    elif NbP==4:
        newax = fig1.add_axes([0.843,0.13,0.05,0.05], anchor='SW', zorder=10)
        
    newax.imshow(im,alpha=alpha)
    newax.axis('off')
    return
    


def FplotmapSI_gp(fig3,ax,data2plot,cmap,norm,plto='tmp_plot',gridpts=True,gridptsgrid=False,gridinc=200,gstyle='lightstyle'): 
    """Plot sea ice map on a geographical projection.
    
    Parameters:
    - fig3: Figure properties.
    - ax: Axis properties.
    - data2plot: Data to plot (xarray).
    - cmap: Colormap for the plot.
    - norm: Normalization for the colormap.
    - plto (str): output plot name.
    - gridpts (bool): Whether to show grid points on axes.
    - gridptsgrid (bool): Whether to show grid points with a grid.
    - gridinc (int): Grid increment.
    - gstyle (str): Grid style ('lightstyle', 'darkstyle', 'ddarkstyle').
    
    Returns:
    tuple: Tuple containing the plot object and the axis object.
    """
    
    cs  = ax.pcolormesh(data2plot,cmap=cmap,norm=norm)

    #ax = plt.gca()
    # Remove the plot frame lines. 
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="off", labeltop='off',left="off", right="off", labelright="off",labelleft="off")  

    
    #if gridpts:
    # show gridpoint on axes
    #    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
    #            labelbottom="off", labeltop='off',left="off", right="off", labelright="off",labelleft="off")  
    #    plto = plto+"_wthgdpts"

    if gridptsgrid:
        lstylegrid=(0, (5, 5)) 
        if (gstyle=='darkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"
            
        if (gstyle=='ddarkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='w'#'#848484'#"#848484"
        if (gstyle=='lightstyle'):
            cmap.set_bad('w')
            lcolorgrid="#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"            

        lalpha=0.2
        lwidthgrid=1.
        #ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(gridinc))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(gridinc))   
        ax.tick_params(colors=tcolorgrid,which="both", bottom=True, top=False,  
                labelbottom=True, labeltop=False,left=True, right=False, labelright=False,labelleft=True)
        ax.grid(which='major',linestyle=lstylegrid,color=lcolorgrid,alpha=lalpha,linewidth=lwidthgrid)
        ax.axhline(y=1.,xmin=0, xmax=883,zorder=10,color=lcolorgrid,linewidth=lwidthgrid,linestyle=lstylegrid,alpha=lalpha )
    
    return cs,ax



    
def Fsaveplt(fig,diro,namo,dpifig=300):
    """Save plot to file.
    
    Parameters:
    - fig: Figure properties to save.
    - diro (str): Output directory.
    - namo (str): Name of the output plot.
    - dpifig (int): Resolution (dpi) of saved plot.
    
    Returns:
    None
   """
    
    fig.savefig(diro+namo+".png", facecolor=fig.get_facecolor(),
                edgecolor='none',dpi=dpifig,bbox_inches='tight', pad_inches=0.01)
    print(diro+namo+".png")
    plt.close(fig) 
    
        
        
def FplotmapARCTIC_gp(fig3,ax,data2plot,cmap,norm,plto='tmp_plot',gridpts=True,gridptsgrid=False,gridinc=200,gstyle='lightstyle'): 
    """Plot Arctic sea ice map on a geographical projection.
    
    Parameters:
    - fig3: Figure properties.
    - ax: Axis properties.
    - data2plot: Data to plot.
    - cmap: Colormap for the plot.
    - norm: Normalization for the colormap.
    - plto (str): Plot name.
    - gridpts (bool): Whether to show grid points on axes.
    - gridptsgrid (bool): Whether to show grid points with a grid.
    - gridinc (int): Grid increment.
    - gstyle (str): Grid style ('lightstyle', 'darkstyle', 'ddarkstyle').
    
    Returns:
    tuple: Tuple containing the plot object and the axis object.
    """
    
    cs  = ax.pcolormesh(data2plot,cmap=cmap,norm=norm)

    #ax = plt.gca()
    # Remove the plot frame lines. 
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="off", labeltop='off',left="off", right="off", labelright="off",labelleft="off")  

    
    if gridpts:
    # show gridpoint on axes
        ax.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", labeltop='off',left="off", right="off", labelright="off",labelleft="on")  
        plto = plto+"_wthgdpts"

    if gridptsgrid:
        lstylegrid=(0, (5, 5)) 
        if (gstyle=='darkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"
            
        if (gstyle=='ddarkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='w'#'#848484'#"#848484"
        if (gstyle=='lightstyle'):
            cmap.set_bad('w')
            lcolorgrid="#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"            

        lalpha=0.2
        lwidthgrid=1.
        #ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(gridinc))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(gridinc))   
        ax.tick_params(axis='x', colors=tcolorgrid)
        ax.tick_params(axis='y', colors=tcolorgrid)
        ax.grid(which='major',linestyle=lstylegrid,color=lcolorgrid,alpha=lalpha,linewidth=lwidthgrid)
        ax.axhline(y=1.,xmin=0, xmax=883,zorder=10,color=lcolorgrid,linewidth=lwidthgrid,linestyle=lstylegrid,alpha=lalpha )
    
    return cs,ax


def Fmycolormap(levbounds,cm_base='Spectral_r',cu='w',co='k',istart=0):
    """Generate a custom colormap based on given level bounds.
    
    Parameters:
    - levbounds (array): [min, max, incr] min, max, and increment of the colormap.
    - cm_base (str): Base colormap name.
    - cu (str): Color for under bounds.
    - co (str): Color for over bounds.
    - istart (int): Starting index for colormap generation.
    
    Returns:
    tuple: Tuple containing the colormap and normalization objects.
    """
    lmin = levbounds[0]
    lmax = levbounds[1]
    incr = levbounds[2]
    levels = np.arange(lmin,lmax,incr)
    if ( (cm_base=='NCL') | (cm_base=='MJO') | (cm_base=='NCL_NOWI') ):
        nice_cmap = slx.make_SLXcolormap(whichco=cm_base)
    else:
        nice_cmap = plt.get_cmap(cm_base)
    colors = nice_cmap(np.linspace(istart/len(levels),1,len(levels)))[:]
    cmap, norm = from_levels_and_colors(levels, colors, extend='max')
    cmap.set_under(cu)
    cmap.set_over(co)
    return cmap,norm

def Fmake_cmap(colors, position=None, bit=False):
    """Make a custom colormap from a list of RGB tuples.
    
    Parameters:
    - colors (list): List of RGB tuples.
    - position (list): List of positions for colors.
    - bit (bool): Whether the RGB values are in 8-bit format (0-255).
    
    Returns:
    colormap: Custom colormap object.
    """
    
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap



def Fmake_SLXcolormap(reverse=False,whichco='MJO',r=0,g=0,b=0):
    """Define a custom colormap.
    
    Parameters:
    - reverse (bool): Whether to reverse the colormap.
    - whichco (str): Which colors to use ('MJO', 'NCL', 'NCL_NOWI').
    - r (int): Red component of a single color.
    - g (int): Green component of a single color.
    - b (int): Blue component of a single color.
    
    Returns:
    colormap: Custom colormap object.
    """

    ### colors to include in my custom colormap
    if whichco=='MJO':
        colors_NCLbipo=[(176,17,3,1),(255,56,8,1),(255,196,1,1),(255,255,255,1),(255,255,255,1),(13,176,255,1),(2,88,255,1),(0,10,174,1)]

    if whichco=='NCL':
        colors_NCLbipo=[(11,76,95),(0,97,128),(0,161,191),(0,191,224),(0,250,250),(102,252,252),(153,250,250),(255,255,255),(255,255,255),(252,224,0),(252,191,0),(252,128,0),(252,64,0),(252,33,0),(128,0,0),(0,0,0)]

    if whichco=='NCL_NOWI':
        colors_NCLbipo=[(11,76,95),(0,97,128),(0,161,191),(0,191,224),(0,250,250),(102,252,252),(153,250,250),(255,255,255),(252,224,0),(252,191,0),(252,128,0),(252,64,0),(252,33,0),(128,0,0),(0,0,0)]
        
    if whichco=='bluyello':
        colors_NCLbipo=[(11,76,95),(0,97,128),(0,161,191),(0,191,224),(0,250,250),(102,252,252),(153,250,250),(255,255,255),(252,224,0),(252,191,0),(252,128,0),(252,64,0),(252,33,0),(128,0,0),(0,0,0)]
        
    if whichco=='oneco':
        colors_NCLbipo=[(r,g,b),(0,0,0)]

    ### Call the function make_cmap which returns my colormap
    my_cmap_NCLbipo = make_cmap(colors_NCLbipo[:], bit=True)
    my_cmap_NCLbipo_r = make_cmap(colors_NCLbipo[::-1], bit=True)
    
    if reverse==True:
        my_cmap_NCLbipo = my_cmap_NCLbipo_r

    return(my_cmap_NCLbipo)

def Fplotrst(datrst,nbb,var2plt,meshmask,it,avecat=False,inum=0,ratio=True,var2pltref=0,pltshow=True,logo=True,pltsave=True,diro='./',ty='T',varty=1,alphalogo=0.3,dpifig=300,cbar=True,sicol='r',sicolwdth=2,pltzoom=False,x1=0,x2=0,x3=0,x4=0,rectcol='r',vmax=10,vmin=0,cblev=[0],linnorm=True,Lzoom=False,zoom=[0,0,0,0],textco='w',pltzone=False,distmask=0,suffix="",cmapforced="no"):
        """Plot map of sea ice restart file

        Parameters:
        - var2plt (str): Variable to plot.
        - it (int): Time index.
        - pltshow (bool): Whether to display the plot.
        - logo (bool): Whether to add Datlas logo.
        - pltsave (bool): Whether to save the plot.
        - varty (int): Variable type (1: ice, 2: ocean, 3: ABL).
        - alphalogo (float): Alpha value for logo.
        - dpifig (int): Resolution (dpi) of the saved plot.
        - cbar (bool): Whether to add a colorbar.
        - sicol (str): Color for sea ice contours.
        - pltzoom (bool): Whether to zoom the plot.
        - x1, x2, x3, x4 (int): Coordinates for zoomed area.
        - rectcol (str): Color for zoom rectangle.

        Returns:
        - fig1: Figure object.
        - ax: Axis object.
        - cs: Contour plot object.
        - cb: Colorbar object.
        """
        
        # get mask
        mask = meshmask.tmask[0,0,:,:]
            
        # get time size
        NT=datrst.time_counter.size
        # date
        tdate=datrst.time_counter.to_index()[it]
    
        
        # get data to plot 

        if 'numcat' in datrst[var2plt].dims:
            if avecat:
                data2plot = datrst[var2plt].isel(time_counter=it).sum(dim='numcat').where(mask!=0)
            else:
                data2plot = datrst[var2plt].isel(time_counter=it,numcat=inum).where(mask!=0)
            
        else:
            data2plot = datrst[var2plt].isel(time_counter=it).where(mask!=0)
            # text label near colorbar
        
        tlabel=var2plt
        

        # plot title
        mb = datrst.attrs.get('mb')
        corrlen = datrst.attrs.get('corrlen')
        std = datrst.attrs.get('std')
        if mb==0:
             titleplt="RST unperturbed"
        else:
            titleplt="RST_mb"+str(mb).zfill(3)+" (l"+str(corrlen)+" std"+str(std)+")"
        print(titleplt)

        if (cmapforced != "no"):
            extend="both"
        else:
            extend="no"
    
        cmap,norm,cblev = Fsetcmapnorm(var2plt,vmin=vmin,vmax=vmax,cblev=cblev,linnorm=linnorm)

        if (cmapforced != "no"):
            cmap = copy.copy(cm.get_cmap(cmapforced))
            cmap.set_under('cyan')
            cmap.set_over('orange')
            

        # main plot
        fig1,(ax) = plt.subplots(1, 1, figsize=[9, 9],facecolor='w')

        if Lzoom:
            gridinc=20   
            # name of plot to output
            namo="ZOOM_"+var2plt+"_RST_mb"+str(mb).zfill(3)+"_l"+str(corrlen)+"_std"+str(std)+"_"+str(it).zfill(4)+"-"+suffix   
        else:
            gridinc=100
            # name of plot to output
            namo="RST_"+var2plt+"_mb"+str(mb).zfill(3)+"_l"+str(corrlen)+"_std"+str(std)+"_"+str(it).zfill(4)+"-"+suffix
        
            
        cs,ax = FplotmapSI_gp(fig1,ax,data2plot,cmap,norm,plto='tmp_plot',gridpts=True,gridptsgrid=True,gridinc=gridinc,gstyle='darkstyle')

        if Lzoom:    
            plt.xlim(zoom[0],zoom[1])
            plt.ylim(zoom[2],zoom[3])
            
            # add date on plot
            tcolordate=textco #"#848484"
            tsizedate=14
            ax.annotate(tdate,xy=(110,502),xycoords='data', color=tcolordate,size=tsizedate)

            # add title exp
            ax.annotate(titleplt,xy=(110,530),xycoords='data', color=tcolordate,size=tsizedate*1.3) 
            
            if pltzone:
                cs3   = ax.contour(distmask.tmask,alpha=0.7,colors='k',linestyles="--",linewidths=1.5,levels=np.arange(1.,2,1.5))
        else:
            # add date on plot
            tcolordate=textco #"#848484"
            tsizedate=14
            ax.annotate(tdate,xy=(15,550),xycoords='data', color=tcolordate,size=tsizedate)

            # add title exp
            ax.annotate(titleplt,xy=(15,520),xycoords='data', color=tcolordate,size=tsizedate*1.3)         

            # add Datlas logo
            if logo:
                FaddDatlasLogo(fig1,alpha=alphalogo)

            if pltzoom:
                rect = patches.Rectangle((x1, x2), x3, x4, linewidth=1, edgecolor=rectcol, facecolor='none',zorder=20)
                ax.add_patch(rect)
                
            if pltzone:
                cs3   = ax.contour(distmask.tmask,alpha=0.7,colors='k',linestyles="--",linewidths=1.5,levels=np.arange(1.,2,1.5))
        
        
        # add colorbar
        if cbar:
            cb = Fpltcolorbar(fig1,ax,var2plt,norm,cmap,cblev,tlabel,textco=textco,F4P=False,extend=extend)   

        if pltshow:
            plt.show()

        if pltsave:    
        # Save fig in png, resolution dpi    
            Fsaveplt(fig1,diro,namo,dpifig=dpifig)
            plt.close(fig1)
            
        return fig1,ax,cs,cb

def Fplot3panels(d1, d2, var2plt, it, compdiff=False, pltshow=True, logo=True, pltsave=True, diro='./', 
                 ty='T', varty=1, maskoce=False, alphalogo=0.3, dpifig=300, cbar=True, sicol='r', 
                 sicolwdth=1, cblev=[0], cblev2=[0], pltzoom=False, x1=0, x2=0, x3=0, x4=0, 
                 rectcol='r', vmax=0.02, vmin=0, vmax2=0.02, vmin2=0, cmapforced="no", 
                 cmapforced2="no", Lzoom=False, zoom=[0, 0, 0, 0], textco='k', pltzone=False, 
                 distmask=0, suffix=""):
    """
    Plot 3-panel figures with 3 maps: d1, d2, and the difference between d1 and d2.
    """

    # Get the mask based on the type
    if ty == 'U':
        mask = d1.meshmask.umask[0, 0, :, :]
    elif ty == 'V':
        mask = d1.meshmask.vmask[0, 0, :, :]
    elif ty == 'F':
        mask = d1.meshmask.fmask[0, 0, :, :]
    else:
        mask = d1.meshmask.tmask[0, 0, :, :]

    # Get ice contour from sea ice concentration
    varsice = 'siconc'
    datasets = [d1, d2]
    icedat = [ds.datice[varsice].isel(time_counter=it).where(mask != 0) for ds in datasets]
    tdate = d1.datice.time_counter.to_index()[it].strftime('%Y-%m-%d %H:%M')
    #tdate  = predate.strftime('%Y-%m-%d %H:%M')
    comptime = compute_days_since_initialtime(d1, it)[1]

    # Mask ocean if required
    if maskoce:
        masks = [ds.datice['siconc'].isel(time_counter=it).where(mask != 0) for ds in datasets]

    # Get data to plot based on varty
    if varty == 1:
        data2plot = [ds.datice[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
        tlabel = d1.datice[var2plt].long_name + " (" + d1.datice[var2plt].units + ")"
    elif varty == 2:
        data2plot = [ds.datoce[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
        tlabel = d1.datoce[var2plt].long_name + " (" + d1.datoce[var2plt].units + ")"
    elif varty == 3:
        data2plot = [ds.databl[var2plt].isel(time_counter=it).where(mask != 0) for ds in datasets]
        tlabel = d1.databl[var2plt].long_name + " (" + d1.databl[var2plt].units + ")"

    # Apply mask if required
    if maskoce:
        data2plot = [data.where(mask > 0.15) for data, mask in zip(data2plot, masks)]

    # Compute the difference if required
    data_difference = data2plot[1] - data2plot[0] if compdiff else None

    # Set grid increment and plot name based on zoom
    gridinc = 50 if Lzoom else 100
    if compdiff:
        suffix=suffix+"_DIFF"
    namo = "3panZOOM_" + d1.prefix + d1.freq + "_" + var2plt+ "_" + suffix  + "_" + str(it).zfill(4) if Lzoom else \
           "3pan_" + d1.prefix + d1.freq + "_" + var2plt + "_" + suffix + "_" + str(it).zfill(4)

    # Plot titles
    titles = [
        d1.frc[0:3] + "+" + d1.rheol,
        d2.frc[0:3] + "+" + d2.rheol,
        "Difference"
    ]

    # Set colormap and normalization for d1 and d2
    cmap, norm, cblev = Fsetcmapnorm(var2plt, vmin=vmin, vmax=vmax, cblev=cblev)
    
    if (cmapforced != "no"):
        cmap = copy.copy(cm.get_cmap(cmapforced))

    # Set colormap and normalization for the difference
    cmap_diff, norm_diff, cblev_diff = Fsetcmapnorm(var2plt, vmin=vmin2, vmax=vmax2, cblev=cblev2)
    if (cmapforced2 != "no"):
        cmap_diff = copy.copy(cm.get_cmap(cmapforced2))
        cmap_diff.set_under('cyan')
        cmap_diff.set_over('orange')

    # Main plot
    fig1, ax = plt.subplots(1, 3, figsize=[18, 4], facecolor='w', gridspec_kw={'width_ratios': [1, 1, 1]})

    # Plot the data
    for i, (a, data, ice) in enumerate(zip(ax, data2plot + [data_difference], icedat + [None])):
        if i < 2:  # For d1 and d2
            cs, a = FplotmapSI_gp(fig1, a, data, cmap, norm, plto='tmp_plot', gridpts=True, 
                                     gridptsgrid=True, gridinc=gridinc, gstyle='darkstyle')
            cs_ice = a.contour(ice, alpha=0.9, colors=sicol, linestyles="-", linewidths=sicolwdth,
                               levels=np.arange(0.15, 0.16, 0.15))
            #a.set_title(titles[i])
        else:  # For the difference
            cs_diff, a = FplotmapSI_gp(fig1, a, data, cmap_diff, norm_diff, plto='tmp_plot', 
                                           gridpts=True, gridptsgrid=True, gridinc=gridinc, 
                                           gstyle='darkstyle')
            #a.set_title(titles[i])
            if ice is not None:
                cs_ice_diff = a.contour(ice, alpha=0.9, colors=sicol, linestyles="-", linewidths=sicolwdth,
                                        levels=np.arange(0.15, 0.16, 0.15))

    # Zoom settings and annotations
    if Lzoom:
        for a in ax:
            a.set_xlim([zoom[0], zoom[1]])
            a.set_ylim([zoom[2], zoom[3]])
            a.annotate(tdate, xy=(105, 510), xycoords='data', color="#dfdfdf", size=10)
            a.annotate("T0 + "+str(comptime)+" days", xy=(105, 500), xycoords='data', color="#dfdfdf", size=10)
            a.annotate(titles[ax.tolist().index(a)], xy=(250, 530), xycoords='data', color="#dfdfdf", size=12)
    else:
        for a in ax:
            a.annotate(tdate, xy=(10, 520), xycoords='data', color="#dfdfdf", size=10)
            a.annotate("T0 + "+str(comptime)+" days", xy=(10, 490), xycoords='data', color="#dfdfdf", size=10)
            a.annotate(titles[ax.tolist().index(a)], xy=(380, 520), xycoords='data', color='w', size=12)

    # Add colorbars
    if cbar:
        barco = 'k' if Lzoom else textco
        cb1 = Fpltcolorbar(fig1, ax[0], var2plt, norm, cmap, cblev, tlabel, F4P=True, textco=barco)
        cb2 = Fpltcolorbar(fig1, ax[1], var2plt, norm, cmap, cblev, tlabel, F4P=True, textco=barco)
        cb_diff = Fpltcolorbar(fig1, ax[2], var2plt, norm_diff, cmap_diff, cblev_diff, tlabel, 
                                   F4P=True, textco=barco,extend="both")

    
    # Add logo
    if logo:
        FaddDatlasLogo(fig1, alpha=alphalogo, NbP=4)

    # Plot rectangle if required
    if pltzoom:
        for a in ax:
            rect = patches.Rectangle((x1, x2), x3, x4, linewidth=1, edgecolor=rectcol, facecolor='none', zorder=20)
            a.add_patch(rect)

    # Plot zone contour if required
    if pltzone:
        for a in ax:
            a.contour(distmask.tmask, alpha=0.7, colors='k', linestyles="--", linewidths=1.5, levels=np.arange(1., 2, 1.5))

    # Adjust subplot spacing
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4)

    # Show and/or save the figure
    if pltshow:
        plt.show()
    if pltsave:
        Fsaveplt(fig1, diro, namo, dpifig=dpifig)
        plt.close(fig1)

    # Return the figure and axes
    return fig1, ax  # Ensure this line is present




def compute_days_since_initialtime(d1, it):
    """
    Compute the number of days since the initial date.

    Parameters:
    d1: The object containing the time data, expected to have a method `datice.time_counter.to_index()`.
    it: The index to get the specific date from the time data.

    Returns:
    tuple: A tuple containing:
        - The number of days since the initial date with full precision.
        - The number of days since the initial date formatted to three decimal places.
    """
    
    # Extract the dates from the datice time_counter
    dates = d1.datice.time_counter.to_index()
    
    # Get the specific date and the initial date
    tdate = dates[it]
    initial_date = dates[0]
    
    # Calculate the time difference
    time_difference = tdate - initial_date
    
    # Calculate the number of days since the initial date with full precision
    days_since_initial_full = time_difference.total_seconds() / (24 * 3600)
    
    # Format the days since initial date to three decimal places
    days_since_initial_3f = f"{days_since_initial_full:.3f}"
    
    return days_since_initial_full, days_since_initial_3f

def Ffillnacorrection(dat):
    """Fill NaN values in data with a specified placeholder value.
    
    Parameters:
    - dat: Input data array.
    
    Returns:
    array: Data array with NaN values filled.
    """
    dat = dat.where(dat!=0.,-9999.)
    dat = dat.fillna(-9999.)
    dat = dat.where(dat!=-9999.)
    return dat

def Ffindinputexpe(prefix,exp,freq,fitype='icemod',diribase="/gpfsstore/rech/cli/regi915/NEMO/NANUK4/"):
    """Find input experiment files.
    
    Parameters:
    - prefix (str): Prefix for experiment files.
    - exp (str): Experiment name.
    - freq (str): Frequency of the experiment.
    - fitype (str): File type ('icemod', 'grid_T', 'ABL').
    - diribase (str): Base directory for experiment files.
    
    Returns:
    tuple: Tuple containing directory and filename patterns.
    """
    allfiles=prefix+exp+"_"+freq+"_*_"+fitype+".nc"
    diri = diribase+prefix+exp+"-S/*/"
    return diri,allfiles 

def FCddef(nbexp,rheol='EVP',frc='BBM',Cddefault=0):
    if Cddefault!=0:
        Cd = Cddefault
    else:
        if nbexp=="600":
            Cd=1.2
        elif nbexp=="701":
            Cd=1.4
        elif nbexp=="702":
            Cd=2
        elif nbexp=="801":
            if rheol=="EVP":
                Cd=1.4
            else:
                Cd=2
        elif nbexp=="802":
            if rheol=="BBM":
                Cd=1.4
            else:
                error_msg2 = "Error: this experiment should not exist ! Check again"
                raise ValueError(error_msg2)
        elif ((nbexp=="903") or (nbexp=="")):
            if rheol=="EVP":
                Cd=1.4
            else:
                Cd=2
        elif nbexp=="904":
            if rheol=="BBM":
                Cd=1.4
            else:
                raise ValueError(error_msg2)
        else:
            error_msg = "Error: unknown Cd value. Please specify as Cddefault=X with X a non zero value)"
            raise ValueError(error_msg)
    return Cd



#-----------------------------------------    
#-----------------------------------------    
class expe:
    """The expe class is a class to load and process a given NEMO simulation
    """

    def __init__(self, diribase, frc,rheol,nb,loadall=False,loadice=False,loadoce=False,loadabl=False,prefix="NANUK4_ICE_ABL-",freq='1d',d1='',d2='',loadmeshmask=True,dirigrid='/gpfsstore/rech/cli/regi915/NEMO/NANUK4/NANUK4.L31-I/',mafi="mesh_mask_NANUK4_L31_4.2.nc",Cddefault=0):
        """Initialize the expe class.

        Parameters:
        - diribase (str): Base directory for experiment files.
        - frc (str): Forcing description.
        - rheol (str): Rheology description.
        - nb (str): Experiment number description.
        - loadall (bool): Whether to load all data.
        - loadice (bool): Whether to load ice data.
        - loadoce (bool): Whether to load ocean data.
        - loadabl (bool): Whether to load ABL data.
        - prefix (str): Prefix for experiment files.
        - freq (str): Frequency of the experiment.
        - d1 (str): Start date for data selection.
        - d2 (str): End date for data selection.
        - loadmeshmask (bool): Whether to load the mesh mask.
        - dirigrid (str): Directory for grid files.
        - mafi (str): Mesh mask filename.

        Returns:
        - An instance of the expe class.
        """
        self.diribase = diribase
        self.prefix = prefix
        self.namexp  = frc+rheol+nb
        self.frc  = frc
        self.rheol = rheol
        self.nbexp    = nb
        self.freq    = freq
        self.d1 = d1
        self.d2 = d2
        self.dirigrid = dirigrid
        self.Cd = FCddef(self.nbexp,self.rheol,self.frc,Cddefault)
        
        print("===== preapring to load experiment: "+self.namexp)
        
        if loadall:
            loadice=True
            loadoce=True
            loadabl=True
            self.Floaddata(rice=loadice,roce=loadoce,rabl=loadabl)
        else:
            if (loadice):
                self.Floaddata(rice=loadice,roce=False,rabl=False)
            if (loadoce):
                self.Floaddata(rice=False,roce=loadoce,rabl=False)
            if (loadabl):
                self.Floaddata(rice=False,roce=False,rabl=loadabl)
            
        if loadmeshmask:
            self.Floadmask(mafi)
            
        
        
        
    def Floaddata(self,rice=True,roce=False,rabl=False,fityi='icemod',fityo='grid_T',fitya='ABL'):
        """Load data of the expe instance.

        Parameters:
        - self (instance of expe class)

        Returns:
        - Modified instance of the expe class.
        """
        if rice:
            self.dirii,self.allfilesi = Ffindinputexpe(self.prefix,self.namexp,self.freq,fitype=fityi,diribase=self.diribase)
            print("Loading ice files : "+self.dirii+self.allfilesi)
            self.datice = xr.open_mfdataset(self.dirii+self.allfilesi,decode_times=True)
            
            
        if roce:
            self.dirio,self.allfileso = Ffindinputexpe(self.prefix,self.namexp,self.freq,fitype=fityo,diribase=self.diribase)
            print("Loading oce files : "+self.dirio+self.allfileso)
            self.datoce = xr.open_mfdataset(self.dirio+self.allfileso,decode_times=True)
            
            
        if rabl:
            self.diria,self.allfilesa = Ffindinputexpe(self.prefix,self.namexp,self.freq,fitype=fitya,diribase=self.diribase)
            print("Loading abl files : "+self.diria+self.allfilesa)
            self.databl = xr.open_mfdataset(self.diria+self.allfilesa,decode_times=True)
            

        return self

    def Floadmask(self,mafi):
        """Load mesh mask.

        Parameters:
        - mafi (str): Mesh mask filename.

        Returns:
        - Modified instance of the expe class.
        """
        self.meshmask = xr.open_dataset(self.dirigrid+mafi,decode_times=True)
        return self
        
        
    def Fplot(self,var2plt,it,ratio=True,var2pltref=0,pltshow=True,logo=True,pltsave=True,maskoce=False,threshmask=0.1,diro='./',ty='T',varty=1,alphalogo=0.3,dpifig=300,cbar=True,sicol='no',sicolwdth=2,pltzoom=False,x1=0,x2=0,x3=0,x4=0,rectcol='r',vmax=10,vmin=0,cblev=[0],cmapforced = "no",Lzoom=False,zoom=[0,0,0,0],textco='w',pltzone=False,distmask=0):
        """Plot map

        Parameters:
        - var2plt (str): Variable to plot.
        - it (int): Time index.
        - pltshow (bool): Whether to display the plot.
        - logo (bool): Whether to add Datlas logo.
        - pltsave (bool): Whether to save the plot.
        - ty (str): Type of data ('T', 'U', 'V', 'F').
        - varty (int): Variable type (1: ice, 2: ocean, 3: ABL).
        - alphalogo (float): Alpha value for logo.
        - dpifig (int): Resolution (dpi) of the saved plot.
        - cbar (bool): Whether to add a colorbar.
        - sicol (str): Color for sea ice contours.
        - pltzoom (bool): Whether to zoom the plot.
        - x1, x2, x3, x4 (int): Coordinates for zoomed area.
        - rectcol (str): Color for zoom rectangle.

        Returns:
        - fig1: Figure object.
        - ax: Axis object.
        - cs: Contour plot object.
        - cs2: Sea ice contour plot object.
        - cb: Colorbar object.
        """
        
        # get mask
        if ty=='U':
            mask = self.meshmask.umask[0,0,:,:]
        if ty=='V':
            mask = self.meshmask.vmask[0,0,:,:]
        if ty=='F':
            mask = self.meshmask.fmask[0,0,:,:]
        else:
            mask = self.meshmask.tmask[0,0,:,:]
            
        #  get ice contour from sea ice concentration
        varsice='siconc'
        icedat = self.datice[varsice].isel(time_counter=it).where(mask!=0)
        NT=self.datice.time_counter.size
        # data to plot 
        tdate=self.datice.time_counter.to_index()[it]

        
        
        # get data to plot (varty says if variable is to read from ice, oce or abl source)
        if varty==1:
            data2plot = self.datice[var2plt].isel(time_counter=it).where(mask!=0)
            # text label near colorbar
            tlabel=self.datice[var2plt].long_name+" ("+self.datice[var2plt].units+")"
        elif varty==2:
            data2plot = self.datoce[var2plt].isel(time_counter=it).where(mask!=0)
            # text label near colorbar
            tlabel=self.datoce[var2plt].long_name+" ("+self.datoce[var2plt].units+")"
        elif varty==3:
            data2plot = self.databl[var2plt].isel(time_counter=it).where(mask!=0)
            # text label near colorbar
            tlabel=self.databl[var2plt].long_name+" ("+self.databl[var2plt].units+")"
            
         # Apply mask if required
        if maskoce:
            data2plot = data2plot.where(icedat > threshmask,-999.)

        # plot title
        titleplt=self.frc[0:3]+"+"+self.rheol
        
        
        cmap,norm,cblev = Fsetcmapnorm(var2plt,vmin=vmin,vmax=vmax,cblev=cblev)

        if (cmapforced != "no"):
            cmap = copy.copy(cm.get_cmap(cmapforced))



        # main plot
        fig1,(ax) = plt.subplots(1, 1, figsize=[12, 12],facecolor='w')

        if Lzoom:
            gridinc=20   
            # name of plot to output
            namo="ZOOM_"+self.prefix+self.namexp+"_"+self.freq+"_"+var2plt+"_"+str(it).zfill(4)   
        else:
            gridinc=100
            # name of plot to output
            namo=self.prefix+self.namexp+"_"+self.freq+"_"+var2plt+"_"+str(it).zfill(4)       
        
            
        cs,ax = FplotmapSI_gp(fig1,ax,data2plot,cmap,norm,plto='tmp_plot',gridpts=True,gridptsgrid=True,gridinc=gridinc,gstyle='darkstyle')
        if sicol != "no":
            cs2   = ax.contour(icedat,alpha=0.9,colors=sicol,linestyles="-",linewidths=sicolwdth,levels=np.arange(threshmask,threshmask+0.01,threshmask))
        else:
            cs2 = None
             
        if Lzoom:    
            plt.xlim(zoom[0],zoom[1])
            plt.ylim(zoom[2],zoom[3])
            
            # add date on plot
            tcolordate=textco #"#848484"
            tsizedate=14
            #ax.annotate(tdate,xy=(260,460),xycoords='data', color=tcolordate,size=tsizedate)
            ax.annotate(tdate,xy=(110,502),xycoords='data', color=tcolordate,size=tsizedate)

            # add title exp
            #ax.annotate(titleplt,xy=(190,475),xycoords='data', color=tcolordate,size=tsizedate*1.3) 
            ax.annotate(titleplt,xy=(110,530),xycoords='data', color=tcolordate,size=tsizedate*1.3) 
            
            if pltzone:
                cs3   = ax.contour(distmask.tmask,alpha=0.7,colors='k',linestyles="--",linewidths=1.5,levels=np.arange(1.,2,1.5))
        else:
            # add date on plot
            tcolordate=textco #"#848484"
            tsizedate=14
            ax.annotate(tdate,xy=(15,550),xycoords='data', color=tcolordate,size=tsizedate)

            # add title exp
            ax.annotate(titleplt,xy=(15,520),xycoords='data', color=tcolordate,size=tsizedate*1.3)         

            # add Datlas logo
            if logo:
                FaddDatlasLogo(fig1,alpha=alphalogo)

            if pltzoom:
                rect = patches.Rectangle((x1, x2), x3, x4, linewidth=1, edgecolor=rectcol, facecolor='none',zorder=20)
                ax.add_patch(rect)
                
            if pltzone:
                cs3   = ax.contour(distmask.tmask,alpha=0.7,colors='k',linestyles="--",linewidths=1.5,levels=np.arange(1.,2,1.5))
        
        
        # add colorbar
        if cbar:
            cb = Fpltcolorbar(fig1,ax,var2plt,norm,cmap,cblev,tlabel,textco=textco,F4P=False)   

        if pltshow:
            plt.show()

        if pltsave:    
        # Save fig in png, resolution dpi    
            Fsaveplt(fig1,diro,namo,dpifig=dpifig)
            plt.close(fig1)

        if cs2 is not None:
            return fig1, ax, cs, cs2, cb
        else:
            return fig1, ax, cs, cb        

            

        

    