"""My SEA ICE tools
A collection of functions i often use when analysing sea ice model outputs (NEMO based). 
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

    
def FaddDatlasLogo(fig1,alpha=0.3,path='/lustre/fswork/projects/rech/cli/commun/misc/logo-datlas-RVB-blanc.png',NbP=1):
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

def Fsetcmapnorm(var2plt,vmin=-1,vmax=1,cblev=[0],linnorm=True):
    """ Set some predefined norm and cmap for some sea ice variables  
    
    Parameters:
    - var2plt: Name of variable to plot.
    - vmin, vmax: min and max of norm to use
    - cblev: ticks to use on colorbar
    - linnorm: set to true to force linear colormap, ortherwise some variables are predefined as non-linear (like concentration)
    
    Returns:
    cmap, norm,cblev
    """
            
    # color map field 
    if ((var2plt=='siconc' or var2plt=='a_i') or var2plt=='a_ip'):
            
            if len(cblev)==1:
                cblev=[0, 0.7,0.8,0.85,0.9,0.95,0.99]
            cmap = copy.copy(cmocean.cm.ice)
            cmap.set_bad('r',1.)
            cmap.set_under('k')
            cmap.set_over('w')
            norm = mcolors.PowerNorm(gamma=4.5,vmin=vmin, vmax=vmax)

        
            #if linnorm: 
            #    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
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

            



if __name__ == "__main__":
    main()


