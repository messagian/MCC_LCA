import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from paretoset import paretoset
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

#Universal gas constant
R = 8.31446261815324 #J/mol/K

def to_zero(num):
    """
    Cutoff function for small floating point numbers. It also acts as an
    assurance for non-negativity

    Parameters
    ----------
    num : float
        Number to evaluate.

    Returns
    -------
    float
        Zero or the number itself.

    """
    if num <= 1e-12:
        return 0.0
    else:
        return num

def extract_last_cycle(res):
    """
    Function to extract the last cycle of the Aspen simulation. The last
    cycle represents the steady state of the system.

    Parameters
    ----------
    res : str
        Tab separated results file, see make_and_run_sims.py

    Returns
    -------
    pd.DataFrame
        The part of the data that contains the last simulation cycle.
    steptime : float
        The size of the timestep of the simulation.

    """
    df = pd.read_csv(res, sep='\t')

    feed_f = list(df.FEED_F)

    feed_f = [to_zero(num) for num in feed_f][:-5]

    dfeedf = np.array(feed_f[:-1]) - np.array(feed_f[1:])

    offset = 5

    for i, el in enumerate(reversed(dfeedf)):
        if el < -1e-10:
            offset = i
            break

    for el in list(reversed(feed_f))[offset:]:
        if el == 0:
            break
        else:
            offset += 1

    offset = len(dfeedf) - offset + 1

    steptime = os.path.dirname(res)[:-1]
    steptime = float(steptime.split('_')[-2])

    return df.iloc[offset:], steptime


def get_segment(df, t_start, t_end):
    """
    Function to extract a portion of the df DataFrame based on an initiial time
    and an end time.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    t_start : float
        Initiial time.
    t_end : float
        End time.

    Returns
    -------
    tmp_df : pd.DataFrame
        The reduced dataframe.

    """
    tmp_df = df[df.Time >= min(df.Time)+t_start]
    tmp_df = tmp_df[tmp_df.Time <= min(df.Time)+t_end]
    return tmp_df


def Cp_func(T):
    """
    Function to compute Cp of the simulated system.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Cp.

    """
    t = T /1000
    A=24.99735
    B=55.18696
    C=-33.69137
    D=7.948387
    E=-0.136638
    return A + B*t + C*t**2+ D*t**3+ E/t**2

def power_comp(mean_stor_F, P_in, P_out, eta_iso=0.7):
    """
    Function to compute the power contribution of the storage flux.

    Parameters
    ----------
    mean_stor_F : float
        Mean storage flux for CO2.
    P_in : float
        Pressure IN.
    P_out : float
        Pressure OUT.
    eta_iso : float, optional
        Efficiency. The default is 0.7.

    Returns
    -------
    float
        Power contribution of the storage flux (W).

    """
    Tin = 273.15 + 30
    Cp = Cp_func(Tin)
    Cv = Cp-R
    k = Cp/Cv
    z = 0.725
    eta_mec = 0.95
    const = 1000*z*k*R*Tin/((k-1)*eta_iso*eta_mec)
    const = const*((P_out/P_in)**((k-1)/k) - 1)

    return const*mean_stor_F

def extra_fuel(power):
    """
    Estimate of extra fuel consumption.

    Parameters
    ----------
    power : float
        Power required.

    Returns
    -------
    float
        Extra fuel consumption (kg).

    """
    lhv = 45.6*1e6 #J/kg
    eta_t = 0.38
    return power/(lhv*eta_t)

def tot_power(mean_stor_F, P1, P2, P3, mean_stor_h2o=0.0):
    """
    Total power required by the system.

    Parameters
    ----------
    mean_stor_F : float
        Mean storage flux for CO2.
    P1 : float
        Pressure at point 1 of the model.
    P2 : float
        Pressure at point 2 of the model.
    P3 : float
        Pressure at point 3 of the model.
    mean_stor_h2o : float, optional
        Mean storage flux for H2O. The default is 0.0.

    Returns
    -------
    float
        Total power required (W).

    """
    power_orc = 5.40051201*1000 #W

    power_c1 = power_comp(mean_stor_F, P1, P2)
    power_c2 = power_comp(mean_stor_F, P2, P3)

    if type(P1) == float:
        if P1 <= 0.01:
            eta = 0.3
        else:
            eta = 0.7
    else:
        eta = []
        for el in P1:
            if el <= 0.01:
                eta.append(0.3)
            else:
                eta.append(0.7)
        eta = np.array(eta)

    power_v = power_comp(mean_stor_F, 1.08, P1, eta)
    power_v2 = power_comp(mean_stor_h2o, 1.08, P1, eta)

    return (-power_c1 - power_c2 + power_v + power_v2 + power_orc)

#V_storage = STORAGE_F*TIME_c/((Pstorage/zRT)*eps_s + (1-eps_s)*q_co2*rho_bed)

def V_stor(mean_stor_F):
    """
    Estimation of the required storage volume.

    Parameters
    ----------
    mean_stor_F : float
        Mean storage flux for CO2.

    Returns
    -------
    float
        Volume of the storage tank.

    """
    Pstorage = 45*1.013*1e5 #Pa
    z = 0.725
    Tin = 273.15 + 30 #K
    eps_s = 0.35
    rho_bed = 340 #kg/m3
    time_day = 10*3600 #s
    q_co2 = 47.11363636 #mol/kg
    return 1000*mean_stor_F*time_day/((Pstorage/(z*R*Tin))*eps_s + (1-eps_s)*q_co2*rho_bed)

def V_capture(
        db = 0.33, #m
        hb = 0.66, #m
        n_beds = 4,
        ):
    """
    Volume of the beds.

    Parameters
    ----------
    db : float, optional
        Diameter. The default is 0.33.
    hb : float, optional
        Height. The default is 0.66.
    hb : int, optional
        Number of beds. The default is 4.
    Returns
    -------
    float
        Volume.

    """


    return n_beds*np.pi*hb*(db**2)/4

def V_tot(mean_stor_F):
    """
    Total volume required.

    Parameters
    ----------
    mean_stor_F : float
        Mean storage flux for CO2.

    Returns
    -------
    float
        Total volume required (m3).

    """
    v_storage = V_stor(mean_stor_F)
    v_capt = V_capture()
    v_orc = 1.0 #m3
    return v_storage + v_capt + v_orc

#power =  1/eff * 1000*STORAGE_F * z*k*R/(k-1) * Tin*((Pout/Pin)**((k-1)/k)-1)
    #tot power = power(c) + power(v) - power(orc) [power orc 5.40051201]
    #Cp = A + B*t + C*t**2+ D*t**3+ E/t**2 J/mol/K
    #Cv = Cp-R
    #extra fuel = tot power/(lhv*eta)
    #V_storage = STORAGE_F*TIME_c/((Pstorage/zRT)*eps_s + (1-eps_s)*q_co2*rho_bed)
#V_tot = V_storage + V_caputre + V_orc

def prepare_df_wa(df_loc):
    """
    Function to compute the weighted average of the results given the two
    regimens in which the simulation is run.

    Parameters
    ----------
    df_loc : str
        Location of the summary table generated by summarize().

    Returns
    -------
    df_wa : pd.DataFrame
        Dataframe containing the weighted average of the results between the
        two conditions.

    """
    df = pd.read_csv(df_loc, sep='\t')
    df_wa = df[df['Outcome'] == 'Weighthed average']
    df_wa = df_wa[~df_wa['CO2 purity'].isna()]
    df_wa = df_wa.drop('Feed', axis=1)
    if 'STORAGE water flowrate (kmol/s)' in df.columns:
        df_wa['Total Power'] = tot_power(df_wa['STORAGE flowrate (kmol/s)'], df_wa['STORAGECO2.P'], np.sqrt(df_wa['STORAGECO2.P']*45), [45]*len(df_wa), df_wa['STORAGE water flowrate (kmol/s)'])
    else:
        df_wa['Total Power'] = tot_power(df_wa['STORAGE flowrate (kmol/s)'], df_wa['STORAGE.P'], np.sqrt(df_wa['STORAGE.P']*45), [45]*len(df_wa))
    df_wa['Extra fuel'] = df_wa['Total Power'].apply(extra_fuel)
    df_wa['Storage Volume'] = df_wa['STORAGE flowrate (kmol/s)'].apply(V_stor)
    df_wa['Total Volume'] = df_wa['STORAGE flowrate (kmol/s)'].apply(V_tot)
    df_wa.to_csv(df_loc.replace('summary_table', 'wa_table'), sep='\t', index=False)
    if not os.path.isdir(df_loc.replace('summary_table.tsv', 'figures')):
        os.makedirs(df_loc.replace('summary_table.tsv', 'figures'))
    # print(df_wa)
    return df_wa

def summary_plots3d(df_wa,
                    projectPath,
                    color='Adsorption time',
                    x='STORAGE flowrate (kmol/s)',
                    y='STORAGE.P',
                    z='Total Power',
                    x_fact=1000,
                    y_fact=1,
                    z_fact=1/1000,
                    color_label='Adsorption time [s]',
                    x_label='Total storage flow [mol/s]',
                    y_label='Storage Pressure [bar]',
                    z_label='Total Power [kW]',
                    elev=30,
                    azim=-24,
                    title='MCC system power requirement',
                  ):
    """
    Fuction to create 3D plots for the data present in the weighted average
    data frames.

    Parameters
    ----------
    df_wa : pd.DataFrame
        Weighted average data frame.
    projectPath : str
        Project main directory.
    color : str, optional
        Column to use for the axis color. The default is 'Adsorption time'.
    x : str, optional
        Column to use for the axis x. The default is 'STORAGE flowrate (kmol/s)'.
    y : str, optional
        Column to use for the axis y. The default is 'STORAGE.P'.
    z : str, optional
        Column to use for the axis z. The default is 'Total Power'.
    x_fact : float, optional
        Scaling factor for the variable if required. The default is 1000.
    y_fact : float, optional
        Scaling factor for the variable if required. The default is 1.
    z_fact : float, optional
        Scaling factor for the variable if required. The default is 1/1000.
    color_label : str, optional
        Label for color axis. The default is 'Adsorption time [s]'.
    x_label : str, optional
        Label for x axis. The default is 'Total storage flow [mol/s]'.
    y_label : str, optional
        Label for y axis. The default is 'Storage Pressure [bar]'.
    z_label : str, optional
        Label for z axis. The default is 'Total Power [kW]'.
    elev : float, optional
        Elevation of the 3D view (deg). The default is 30.
    azim : float, optional
        Azimuth of the 3D view (deg). The default is -24.
    title : str, optional
        Title of the plot. The default is 'MCC system power requirement'.

    Returns
    -------
    None.

    """
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    patches = []
    col_dict = {}
    for ads, c in zip(sorted(set(df_wa[color])), CB_color_cycle):
        col_dict[ads] = c
        patches.append(mpatches.Patch(color=col_dict[ads], label=str(ads)))

    col = []
    for ads in df_wa[color]:
        col.append(col_dict[ads])

    ax.scatter(df_wa[x]*x_fact, df_wa[y]*y_fact, df_wa[z]*z_fact, c=col)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.view_init(elev=elev, azim=azim)
    plt.legend(handles=patches, title=color_label, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, ncol=len(patches))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(projectPath, 'figures', f'{title}.tiff'), dpi=250)

def summary_plots2d(df_wa, projectPath,
                    color='Adsorption time',
                    x='STORAGE flowrate (kmol/s)',
                    y='STORAGE.P',
                    x_fact=1000,
                    y_fact=1,
                    color_label='Adsorption time [s]',
                    x_label='Total storage flow [mol/s]',
                    y_label='Storage Pressure [bar]',
                    title='MCC system power requirement',
                    pareto=False,
                    pareto2=False,
                    zoom=False,
                  ):
    """


    Parameters
    ----------
    df_wa : pd.DataFrame
        Weighted average data frame.
    projectPath : str
        Project main directory.
    color : str, optional
        Column to use for the axis color. The default is 'Adsorption time'.
    x : str, optional
        Column to use for the axis x. The default is 'STORAGE flowrate (kmol/s)'.
    y : str, optional
        Column to use for the axis y. The default is 'STORAGE.P'.
    x_fact : float, optional
        Scaling factor for the variable if required. The default is 1000.
    y_fact : float, optional
        Scaling factor for the variable if required. The default is 1.
    color_label : str, optional
        Label for color axis. The default is 'Adsorption time [s]'.
    x_label : Tstr, optional
        Label for x axis. The default is 'Total storage flow [mol/s]'.
    y_label : str, optional
        Label for y axis. The default is 'Storage Pressure [bar]'.
    title : str, optional
        Plot title. The default is 'MCC system power requirement'.
    pareto : bool, optional
        Wether or not to plot the pareto front. The default is False.
    pareto2 : bool, optional
        Wether or not to plot the pareto front (different kind).
            The default is False.
    zoom : bool, optional
        Wether or not to subplot a small area of the data. The default is False.

    Returns
    -------
    None.

    """
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig = plt.figure()
    ax = fig.add_subplot()

    patches = []
    col_dict = {}
    for ads, c in zip(sorted(set(df_wa[color])), CB_color_cycle):
        col_dict[ads] = c
        patches.append(mpatches.Patch(color=col_dict[ads], label=str(ads)))

    col = []
    for ads in df_wa[color]:
        col.append(col_dict[ads])

    ax.scatter(df_wa[x]*x_fact, df_wa[y]*y_fact, c=col, s=5)

    if zoom:

        # x1,x2,y1,y2 = 60.4, 60.9, 89.9, 90.1
        x1,x2,y1,y2 = zoom

        axin = ax.inset_axes([0.5, 0.1, 0.44, 0.47],
            xlim=(x1, x2), ylim=(y1, y2))#, xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        for el in axin.get_xticklabels()+axin.get_yticklabels():
            el.set_fontsize('7')
        axin_ticklabels = axin.get_yticklabels()

        axin.set_yticks([axin.get_yticks()[0], axin.get_yticks()[-1]])
        axin.set_yticklabels((axin_ticklabels[0], axin_ticklabels[-1]))
        axin.set_xticks([65.2, 65.32])
        axin.set_xticklabels(['65.20', '65.32'], ha='right')


        axin.scatter(df_wa[x]*x_fact, df_wa[y]*y_fact, c=col, s=5)
        ax.indicate_inset((x1-0.3, y1-0.1, 0.7, 0.2), axin, edgecolor="black")
        # ax.indicate_inset_zoom(axin, edgecolor="black")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    extra_name = ''
    if pareto:
        mask = paretoset(pd.DataFrame([df_wa[x], df_wa[y]]).T, sense=["max", "max"])
        df_par = df_wa[mask]
        df_par = df_par.sort_values(x)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        x_vals = [0] + list(df_par[x]*x_fact) + [100]
        y_vals = [100] + list(df_par[y]*y_fact) + [0]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.plot(x_vals, y_vals, c='black', zorder=0, linewidth=.5)
        extra_name = '_pareto1'

    elif pareto2:
        mask = paretoset(pd.DataFrame([df_wa[x], df_wa[y]]).T, sense=["max", "max"])
        df_par = df_wa[mask]
        df_par = df_par.sort_values(x)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        x_vals = [-10] + list(df_par[x]*x_fact) + [max(list(df_par[x]*x_fact))]
        y_vals = [max(list(df_par[y]*y_fact))] + list(df_par[y]*y_fact) + [-10]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.plot(x_vals, y_vals, c='black', zorder=0, linewidth=.5)
        extra_name = '_pareto2'

    plt.legend(handles=patches, title=color_label, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, ncol=len(patches))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(projectPath, 'figures', f'{title}_{extra_name}.tiff'), dpi=250)


def computation(res,
                timings=np.array([1, 0.1, 7/30, 37/15]),
                cycles=['ads', 'bld', 'prg', 'heat'],
                time_delta=[0.0, 1.1, 1.0, 1.5, 1.0],
                flowrate_factor=1,
                proj='PPN',
                ):
    """
    Function to extract and prepare the data for plotting and representation.

    Parameters
    ----------
    res : tr
        Tab separated results file, see make_and_run_sims.py
    timings : np.array, optional
        Time proportion of the different steps of the simulation.
            The default is np.array([1, 0.1, 7/30, 37/15]).
    cycles : list, optional
        Names of the cycle steps. The default is ['ads', 'bld', 'prg', 'heat'].
    time_delta : list, optional
        Correction factors for the time data extraction.
            The default is [0.0, 1.1, 1.0, 1.5, 1.0].
    flowrate_factor : float, optional
        Corrective factor for KAUST7. The default is 1.
    proj : str, optional
        Project name. The default is 'PPN'.

    Returns
    -------
    None.

    """

    df, steptime = extract_last_cycle(res)
    for c in df.columns:
        df[c] = df[c].apply(to_zero)

# PPN               adsor       blowd           purge       desor/heat
# kaust7    adsor blowd heat press
# zeolite13x adsor recy heat purge

    timings = steptime*timings

    dfs = {}
    for i, c in enumerate(cycles):
        t_start = sum(timings[:i])
        t_end = sum(timings[:i+1])
        dfs[c] = get_segment(df, t_start+time_delta[i], t_end+time_delta[i+1])

    if proj == 'KAUST7':

        df_heat = df[df['STORAGE_F'].apply(to_zero) != 0.0]
        co2_pur = df_heat['STORAGEY_Fwd("CO2")'].mean()
        tot_flowrate = flowrate_factor*df_heat['STORAGE_F'].mean()
        prod = np.mean(flowrate_factor*df_heat['storage_CO2'])

        ads_val = sum(df['feed_CO2'])
        tail_val = sum(df['tailpipe_CO2'])

        co2_rec = (ads_val - tail_val)/ads_val
        return np.array([co2_pur, tot_flowrate, co2_rec, prod])
    elif proj == 'zeolite':
        # zeolite13x adsor recy heat purge

        df_heat = df[df['STORAGECO2_F'].apply(to_zero) != 0.0]
        co2_pur = df_heat['STORAGECO2Y_Fwd("CO2")'].mean()
        tot_flowrate = flowrate_factor*df_heat['STORAGECO2_F'].mean()
        prod = np.mean(flowrate_factor*df_heat['STORAGECO2_F']*df_heat['STORAGECO2Y_Fwd("CO2")'])

        df_heat_h2o = df[df['STORAGEH2O_F'].apply(to_zero) != 0.0]
        h2o_flowrate = flowrate_factor*df_heat_h2o['STORAGEH2O_F'].mean()

        ads_val = sum(df['FEED_F']*df['FEEDY_Fwd("CO2")'])
        tail_val = sum(df['TAILPIPE_F']*df['TAILPIPEY_Fwd("CO2")'])

        recy_val_co2 = sum(df['ROUTCO2_F']*df['ROUTCO2Y_Fwd("CO2")'])
        recy_val_h2o = sum(df['ROUTH2O_F']*df['ROUTH2OY_Fwd("CO2")'])
        recy_val = recy_val_co2 + recy_val_h2o
        storage_h2o = sum(flowrate_factor*df['STORAGEH2O_F']*df['STORAGEH2OY_Fwd("CO2")'])

        prod = np.mean(flowrate_factor*df['STORAGECO2_F']*df['STORAGECO2Y_Fwd("CO2")'])

        co2_rec = (ads_val - tail_val - recy_val - storage_h2o)/ads_val
        return np.array([co2_pur, tot_flowrate, h2o_flowrate, co2_rec, prod])
    else: #PPN


        ads_val = sum(df['FEED_F']*df['FEEDY_Fwd("CO2")'])
        tail_val = sum(df['TAILPIPE_F']*df['TAILPIPEY_Fwd("CO2")'])
        purge_val = sum(df['RECOUT_F']*df['RECOUTY_Fwd("CO2")'])

        df_heat = df[df['STORAGE_F'].apply(to_zero) != 0.0]
        co2_pur = df_heat['STORAGEY_Fwd("CO2")'].mean()
        tot_flowrate = flowrate_factor*df_heat['STORAGE_F'].mean()
        prod = np.mean(flowrate_factor*df_heat['STORAGE_F']*df_heat['STORAGEY_Fwd("CO2")'])

        co2_rec = (ads_val - tail_val - purge_val)/ads_val

        return np.array([co2_pur, tot_flowrate, co2_rec, prod])

def summarize(projectPath,
              columns=['Outcome',
                       'RECOUT.P',
                       'STORAGE.P',
                       'Adsorption time',
                       'Feed',
                       'CO2 purity',
                       'STORAGE flowrate (kmol/s)',
                       'CO2 recovery',
                       'Productivity (kmol/s)',
                       ],
              n_params=4,
              feed_map={
                  3.617*1e-3: 90,
                  4.138*1e-3: 105,
                },
              computation_kwargs={
                  'timings': np.array([1, 0.1, 7/30, 37/15]),
                  'cycles': ['ads', 'bld', 'prg', 'heat'],
                  'time_delta': [0.0, 1.1, 1.0, 1.5, 1.0],
                  'flowrate_factor': 1,
                  'proj': 'PPN',
              },
              ):
    """
    Function to unify the results of the simulations in an unique text based
    tab separated file

    Parameters
    ----------
    projectPath : str
        Project main directory.
    columns : list
        Columns of the data. The default is
                    ['Outcome',
                     'RECOUT.P',
                     'STORAGE.P',
                     'Adsorption time',
                     'Feed',
                     'CO2 purity',
                     'STORAGE flowrate (kmol/s)',
                     'CO2 recovery',
                     'Productivity (kmol/s)',
                     ]
    n_params : int, optional
        Parameters varied in the simulations. The default is 4.
    feed_map : dict, optional
        Coefficients for the two different regimens of the simulations (k) and
        the identifying speed (v).
        The default is
        {
            3.617*1e-3: 90,
            4.138*1e-3: 105,
          }.
    computation_kwargs : dict, optional
        Keyword arguments to pass to computation(). The default is
            {
                'timings': np.array([1, 0.1, 7/30, 37/15]),
                'cycles': ['ads', 'bld', 'prg', 'heat'],
                'time_delta': [0.0, 1.1, 1.0, 1.5, 1.0],
                'flowrate_factor': 1,
                'proj': 'PPN',
            }.

    Returns
    -------
    None.

    """

    summary = open(os.path.join(projectPath, 'summary_table.tsv'), 'w')
    summary.write('\t'.join(columns)+'\n')
    res_root = os.path.join(projectPath, 'Res')
    final_values = {}

    for res in tqdm(os.listdir(res_root)):
        if 'S_' not in res:
            if 'F_' not in res:
                continue

        outcome, params = res.split('_', 1)
        params = params.split('_')[-n_params:]

        if feed_map[float(params[-1])] == 90:
            weight = 1-0.905263158
        else:
             weight = 0.905263158

        if tuple(params[:-1]) not in final_values:
            final_values[tuple(params[:-1])] = {}

        summary.write('\t'.join([outcome]+params))

        if outcome == 'S':
            results = computation(os.path.join(res_root, res, 'results.tsv'), **computation_kwargs)
            summary.write('\t')
            summary.write('\t'.join(results.astype(str)))
            final_values[tuple(params[:-1])][weight] = results
        else:
            summary.write('\t\t\t\t')
            final_values[tuple(params[:-1])][weight] = None

        summary.write('\n')
        if any(x is None for x in final_values[tuple(params[:-1])].values()) and len(list(final_values[tuple(params[:-1])].values())) == 2:
            summary.write('Weighthed average\t')
            summary.write('\t'.join(tuple(params[:-1])))
            summary.write('\t\t')
            summary.write('\t\t\t')
            summary.write('\n')
        elif len(list(final_values[tuple(params[:-1])].values())) == 2:
            summary.write('Weighthed average\t')
            summary.write('\t'.join(tuple(params[:-1])))
            summary.write('\t\t')
            avg_vals = sum([w*final_values[tuple(params[:-1])][w] for w in final_values[tuple(params[:-1])]])
            summary.write('\t'.join(avg_vals.astype(str)))
            summary.write('\n')
        else:
            pass

    summary.close()


if __name__ == '__main__':
    projectPath = filedialog.askdirectory(initialdir=r"C:\Users\pezzelg\Desktop\universita giuseppe_13_11_2023\PhD\CO2 removal\MCC paper\LCA_auto\projects")

    if not projectPath:
        print('No project selected')
        sys.exit()
    if 'PPN' in projectPath:
        summarize(projectPath,
                columns=['Outcome',
                        'RECOUT.P',
                        'STORAGE.P',
                        'Adsorption time',
                        'Feed',
                        'CO2 purity',
                        'STORAGE flowrate (kmol/s)',
                        #STORAGE water flowrate (kmol/s) only zeolite
                        'CO2 recovery',
                        'Productivity (kmol/s)',
                        ],)
        storage_p_label = 'STORAGE.P'
    elif 'KAUST' in projectPath:
        computation_kwargs={
                  'timings': np.array([1, 0, 1]),
                  'cycles': ['ads', 'bld', 'heat'],
                  'time_delta': [0.0, 1.6, 2.6, 1.6],
                  'flowrate_factor': 2,
                  'proj': 'KAUST7',
        }
        summarize(projectPath,
                columns=['Outcome',
                        'B1.Flowrate (kmol/s)',
                        'STORAGE.P',
                        'Adsorption time',
                        'Feed',
                        'CO2 purity',
                        'STORAGE flowrate (kmol/s)',
                        #STORAGE water flowrate (kmol/s) only zeolite
                        'CO2 recovery',
                        'Productivity (kmol/s)',
                        ],
                  feed_map={
                  3.617*1e-3/2: 90,
                  4.138*1e-3/2: 105,
                },
        computation_kwargs=computation_kwargs,

        )
        storage_p_label = 'STORAGE.P'
    else: #zeolite
        computation_kwargs={
                  'timings': np.array([1, 0.08, 2, 0.08]),
                  'cycles': ['ads', 'recy', 'heat', 'purge'],
                  'time_delta': [0.0, 1.1, 1.6, 1.6, 1.1],
                  'flowrate_factor': 1,
                  'proj': 'zeolite',
        }
        summarize(projectPath,
                  columns=['Outcome',
                       'STORAGEH2O.P',
                       'STORAGECO2.P',
                       'VP.Flowrate',
                       'Adsorption time',
                       'Feed',
                       'CO2 purity',
                       'STORAGE flowrate (kmol/s)',
                       'STORAGE water flowrate (kmol/s)',
                       'CO2 recovery',
                       'Productivity (kmol/s)',
                       ],
                  feed_map={
                  3.748*1e-3: 90,
                  4.296*1e-3: 105,
                },
        computation_kwargs=computation_kwargs,
        n_params=5,
        )
        storage_p_label = 'STORAGECO2.P'

    df_loc = os.path.join(projectPath, 'summary_table.tsv')
    df_wa = prepare_df_wa(df_loc)

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='CO2 purity',
                    x_fact=100,
                    y_fact=100,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='$CO_2$ purity [%]',
                    title='MCC $CO_2$ recovery vs $CO_2$ purity zoom',
                    pareto=True,
                    zoom=(65.2, 65.32, 96.83, 96.86),
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='CO2 purity',
                    x_fact=100,
                    y_fact=100,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='$CO_2$ purity [%]',
                    title='MCC $CO_2$ recovery vs $CO_2$ purity',
                    pareto=True,
                    #zoom=(65.2, 65.32, 96.83, 96.86),
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='CO2 purity',
                    x_fact=100,
                    y_fact=100,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='$CO_2$ purity [%]',
                    title='MCC $CO_2$ recovery vs $CO_2$ purity zoom',
                    pareto2=True,
                    zoom=(65.2, 65.32, 96.83, 96.86),
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='CO2 purity',
                    x_fact=100,
                    y_fact=100,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='$CO_2$ purity [%]',
                    title='MCC $CO_2$ recovery vs $CO_2$ purity',
                    pareto2=True,
                    #zoom=(65.2, 65.32, 96.83, 96.86),
                    )
    # plt.show()
    # sys.exit()

    summary_plots3d(df_wa, projectPath,
                    y=storage_p_label)

    summary_plots3d(df_wa, projectPath,
                    color='Adsorption time',
                    x='CO2 recovery',
                    y=storage_p_label,
                    z='CO2 purity',
                    x_fact=100,
                    y_fact=1,
                    z_fact=100,
                    color_label='Adsorption time [s]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Desorption pressure [bar]',
                    z_label='$CO_2$ purity [%]',
                    title='MCC $CO_2$ purity and recovery',
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='Extra fuel',
                    x_fact=100,
                    y_fact=-100/0.008495831,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Extra fuel [%]',
                    title='MCC Extra fuel requirement',
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='Total Power',
                    x_fact=100,
                    y_fact=-0.001,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Total Power [kW]',
                    title='MCC Total Power requirement',
                    )

    summary_plots2d(df_wa, projectPath,
                    x=storage_p_label,
                    y='CO2 purity',
                    x_fact=1,
                    y_fact=100,
                    x_label='Desorption pressure [bar]',
                    y_label='$CO_2$ purity [%]',
                    title='MCC Pressure vs $CO_2$ purity',
                    )

    summary_plots2d(df_wa, projectPath,
                    x=storage_p_label,
                    y='CO2 recovery',
                    x_fact=1,
                    y_fact=100,
                    x_label='Desorption pressure [bar]',
                    y_label='$CO_2$ recovery [%]',
                    title='MCC Pressure vs $CO_2$ recovery',
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='STORAGE flowrate (kmol/s)',
                    x_fact=100,
                    y_fact=1000,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Total storage flow [mol/s]',
                    title='MCC Flow vs $CO_2$ recovery',
                    )

    summary_plots2d(df_wa, projectPath,
                    x='CO2 recovery',
                    y='Storage Volume',
                    x_fact=100,
                    y_fact=1,
                    x_label='$CO_2$ recovery [%]',
                    y_label='Storage Volume [$m^3$]',
                    title='MCC $CO_2$ recovery vs Storage Volume',
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='Storage Volume',
                    x_fact=100,
                    y_fact=1,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Storage Volume [$m^3$]',
                    title='MCC $CO_2$ recovery vs Storage Volume',
                    )

    summary_plots2d(df_wa, projectPath,
                    x='CO2 recovery',
                    y='Total Volume',
                    x_fact=100,
                    y_fact=1,
                    x_label='$CO_2$ recovery [%]',
                    y_label='Total Volume [$m^3$]',
                    title='MCC $CO_2$ recovery vs Total Volume',
                    )

    summary_plots2d(df_wa, projectPath,
                    color=storage_p_label,
                    x='CO2 recovery',
                    y='Total Volume',
                    x_fact=100,
                    y_fact=1,
                    color_label='Desorption pressure [bar]',
                    x_label='$CO_2$ recovery [%]',
                    y_label='Total Volume [$m^3$]',
                    title='MCC $CO_2$ recovery vs Total Volume',
                    )
    if 'KAUST' in projectPath:
        for p in np.unique(df_wa[storage_p_label]):
            df_tmp = df_wa[df_wa[storage_p_label] == p]
            df_tmp['B1.Flowrate (kmol/s)'] = df_tmp['B1.Flowrate (kmol/s)']*100
            summary_plots2d(df_tmp, projectPath,
                            color='B1.Flowrate (kmol/s)',
                            x='Adsorption time',
                            y='CO2 recovery',
                            x_fact=1,
                            y_fact=100,
                            color_label='Percentage of FEED [%]',
                            x_label='Adsorption time [s]',
                            y_label='$CO_2$ recovery [%]',
                            title=f'$CO_2$ recovery vs Adsorption time P={p} bar',
                            pareto=False,
                            zoom=False,
                        )

            summary_plots2d(df_tmp, projectPath,
                            color='B1.Flowrate (kmol/s)',
                            x='Adsorption time',
                            y='CO2 purity',
                            x_fact=1,
                            y_fact=100,
                            color_label='Percentage of FEED [%]',
                            x_label='Adsorption time [s]',
                            y_label='$CO_2$ purity [%]',
                            title=f'$CO_2$ purity vs Adsorption time P={p} bar',
                            pareto=False,
                            zoom=False,
                        )

    if 'Zeolite' in projectPath:
        for p in np.unique(df_wa[storage_p_label]):
            df_tmp1 = df_wa[df_wa[storage_p_label] == p]
            for vp in np.unique(df_wa['VP.Flowrate']):
                df_tmp = df_tmp1[df_tmp1['VP.Flowrate'] == vp]
                df_tmp['VP.Flowrate'] = df_tmp['VP.Flowrate']*100
                if df_tmp.empty:
                    continue
                summary_plots2d(df_tmp, projectPath,
                                color='Adsorption time',
                                x='STORAGEH2O.P',
                                y='CO2 purity',
                                x_fact=1,
                                y_fact=100,
                                color_label='Adsorption time [s]',
                                x_label='Desorption pressure $H_2O$ [bar]',
                                y_label='$CO_2$ purity [%]',
                                title=f'$CO_2$ purity vs Desorption $P_{{H2O}}$. $P_{{CO2}}$={p} bar; Purge={vp*100} %',
                                pareto=False,
                                zoom=False,
                            )
                # plt.show()
        for p in np.unique(df_wa[storage_p_label]):
            df_tmp1 = df_wa[df_wa[storage_p_label] == p]

            for vp in np.unique(df_wa['VP.Flowrate']):
                df_tmp = df_tmp1[df_tmp1['VP.Flowrate'] == vp]

                df_tmp['VP.Flowrate'] = df_tmp['VP.Flowrate']*100

                if df_tmp.empty:
                    continue

                summary_plots2d(df_tmp, projectPath,
                                color='Adsorption time',
                                x='STORAGEH2O.P',
                                y='CO2 recovery',
                                x_fact=1,
                                y_fact=100,
                                color_label='Adsorption time [s]',
                                x_label='Desorption pressure $H_2O$ [bar]',
                                y_label='$CO_2$ recovery [%]',
                                title=f'$CO_2$ recovery vs Desorption $P_{{H2O}}$. $P_{{CO2}}$={p} bar; Purge={vp*100} %',
                                pareto=False,
                                zoom=False,
                            )
        for p in np.unique(df_wa[storage_p_label]):
            df_tmp1 = df_wa[df_wa[storage_p_label] == p]
            for pW in np.unique(df_wa['STORAGEH2O.P']):
                df_tmp = df_tmp1[df_tmp1['STORAGEH2O.P'] == pW]
                if df_tmp.empty:
                    continue
                summary_plots2d(df_tmp, projectPath,
                                color='Adsorption time',
                                x='VP.Flowrate',
                                y='CO2 recovery',
                                x_fact=100,
                                y_fact=100,
                                color_label='Adsorption time [s]',
                                x_label='Percentage of FEED [%]',
                                y_label='$CO_2$ recovery [%]',
                                title=f'$CO_2$ recovery vs Purge flow. $P_{{CO2}}$={p} bar; $P_{{H2O}}$={pW} bar',
                                pareto=False,
                                zoom=False,
                            )
        for p in np.unique(df_wa[storage_p_label]):
            df_tmp1 = df_wa[df_wa[storage_p_label] == p]
            for pW in np.unique(df_wa['STORAGEH2O.P']):
                df_tmp = df_tmp1[df_tmp1['STORAGEH2O.P'] == pW]
                if df_tmp.empty:
                    continue
                summary_plots2d(df_tmp, projectPath,
                                color='Adsorption time',
                                x='VP.Flowrate',
                                y='CO2 purity',
                                x_fact=100,
                                y_fact=100,
                                color_label='Adsorption time [s]',
                                x_label='Percentage of FEED [%]',
                                y_label='$CO_2$ purity [%]',
                                title=f'$CO_2$ purity vs Purge flow. $P_{{CO2}}$={p} bar; $P_{{H2O}}$={pW} bar',
                                pareto=False,
                                zoom=False,
                            )


    # summary_plots2d(df_wa, df_loc,
    #                 color='',
    #                 x='',
    #                 y='',
    #                 x_fact=,
    #                 y_fact=,
    #                 color_label=,
    #                 x_label=,
    #                 y_label=,
    #                 title=,
    #                 )

    # plt.show()