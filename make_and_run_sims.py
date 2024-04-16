import os
import sys
import shutil
import time
from comtypes import client
import subprocess
from tkinter import filedialog

def create_files_and_folders_PPN6CH2TETA(recout, storage, feed, steptime):
    """
    Fuction to setup and create the files and the folders for the PPN6CH2TETA
    model simulation.

    Parameters
    ----------
    recout : float/str
        Value for the corresponding variable in the Aspen model.
    storage : float/str
        Value for the corresponding variable in the Aspen model.
    feed : float/str
        Value for the corresponding variable in the Aspen model.
    steptime : float/str
        Value for the corresponding variable in the Aspen model.

    Returns
    -------
    res_folder : str
        Folder in which the results are written.
    dest : str
        Location of the modified ada file.

    """

    results_root = os.path.abspath(os.path.join(root, 'Res'))
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    run_name = f'{basename}_{recout}_{storage}_{steptime:03.1f}_{feed[0]}'
    res_folder = os.path.abspath(os.path.join(results_root, run_name))

    if not os.path.isdir(res_folder):
        if os.path.isdir(os.path.join(results_root, f'S_{run_name}')):
            print('DONE S')
            return None
        elif os.path.isdir(os.path.join(results_root, f'F_{run_name}')):
            print('DONE F')
            return None
        os.makedirs(res_folder)

    with open(template_path) as template:
        templ = template.read()

    shutil.copytree(properties_folder, os.path.abspath(os.path.join(res_folder, f'AM_run_{recout}_{storage}_{steptime:03.1f}_{feed[0]}')))

    variables = {
        'recout_p': str(recout),
        'storage_p': str(storage),
        }

    variables['co_step_data2_step_time'] = str(steptime)
    variables['co_step_data3_step_time'] = str(steptime*0.1)
    variables['co_step_data4_step_time'] = str(steptime*7/30)
    variables['co_step_data5_step_time'] = str(steptime*37/15)
    variables['feed_f'] = str(feed[0])
    variables['feed_y_fwd_co2'] = str(feed[1])
    variables['feed_y_fwd_N2'] = str(feed[2])
    variables['bed_mflow'] = str(feed[3])

    cumsum = 0
    for v in variables:
        if 'step' in v:
            cumsum += float(variables[v])
    cumsum /= steptime
    assert cumsum <= 4
    assert cumsum >= 3

    for v in variables:
        templ = templ.replace(v, variables[v])
    dest = os.path.join(root, 'Res', f'{run_name}',
              f'run_{recout}_{storage}_{steptime:03.1f}_{feed[0]}.ada')
    with open(dest, 'w') as o:
        o.write(templ)

    return res_folder, dest

def create_files_and_folders_zeolite13x(stoh, stoc, vph, feed, steptime):
    """
    Fuction to setup and create the files and the folders for the zeolite13x
    model simulation.

    Parameters
    ----------
    stoh : float/str
        Value for the corresponding variable in the Aspen model.
    stoc : float/str
        Value for the corresponding variable in the Aspen model.
    vph : float/str
        Value for the corresponding variable in the Aspen model.
    feed : float/str
        Value for the corresponding variable in the Aspen model.
    steptime : float/str
        Value for the corresponding variable in the Aspen model.

    Returns
    -------
    res_folder : str
        Folder in which the results are written.
    dest : str
        Location of the modified ada file.

    """
    results_root = os.path.abspath(os.path.join(root, 'Res'))
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    run_name = f'{basename}_{stoh}_{stoc}_{vph}_{steptime:03.1f}_{feed[0]}'
    res_folder = os.path.abspath(os.path.join(results_root, run_name))

    if not os.path.isdir(res_folder):
        if os.path.isdir(os.path.join(results_root, f'S_{run_name}')):
            print('DONE S')
            return None
        elif os.path.isdir(os.path.join(results_root, f'F_{run_name}')):
            print('DONE F')
            return None
        os.makedirs(res_folder)

    with open(template_path) as template:
        templ = template.read()

    shutil.copytree(properties_folder, os.path.abspath(os.path.join(res_folder, f'AM_run_{stoh}_{stoc}_{vph}_{steptime:03.1f}_{feed[0]}')))

    variables = {
        'storageh2o_p': str(stoh),
        'storageco2_p': str(stoc),
        }

    variables['t_ads'] = str(steptime)
    variables['t_heat'] = str(steptime*2)
    variables['t_rec'] = str(steptime*0.08)
    variables['t_pur'] = str(steptime*0.04)
    variables['feed_f'] = str(feed[0])
    variables['feed_y_fwd_co2'] = str(feed[1])
    variables['feed_y_fwd_H2O'] = str(feed[2])
    variables['feed_y_fwd_N2'] = str(feed[3])
    variables['bed_mflow'] = str(feed[4])
    variables['vspurgeh2o'] = str(vph*feed[0])

    cumsum = 0
    for v in variables:
        if 't_' in v:
            cumsum += float(variables[v])
    cumsum /= steptime
    assert cumsum <= 4
    assert cumsum >= 3

    for v in variables:
        templ = templ.replace(v, variables[v])
    dest = os.path.join(root, 'Res', f'{run_name}',
              f'run_{stoh}_{stoc}_{vph}_{steptime:03.1f}_{feed[0]}.ada')
    with open(dest, 'w') as o:
        o.write(templ)

    return res_folder, dest

def create_files_and_folders_KAUST7(b1, storage, feed, steptime):
    """
    Fuction to setup and create the files and the folders for the KAUST7
    model simulation.

    Parameters
    ----------
    b1 : float/str
        Value for the corresponding variable in the Aspen model.
    storage : float/str
        Value for the corresponding variable in the Aspen model.
    feed : float/str
        Value for the corresponding variable in the Aspen model.
    steptime : float/str
        Value for the corresponding variable in the Aspen model.

    Returns
    -------
    res_folder : str
        Folder in which the results are written.
    dest : str
        Location of the modified ada file.

    """
    results_root = os.path.abspath(os.path.join(root, 'Res'))
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    run_name = f'{basename}_{b1}_{storage}_{steptime:03.1f}_{feed[0]}'
    res_folder = os.path.abspath(os.path.join(results_root, run_name))

    if not os.path.isdir(res_folder):

        if os.path.isdir(os.path.join(results_root, f'S_{run_name}')):
            print('DONE S')
            return None
        elif os.path.isdir(os.path.join(results_root, f'F_{run_name}')):
            print('DONE F')
            return None
        os.makedirs(res_folder)

    with open(template_path) as template:
        templ = template.read()

    shutil.copytree(properties_folder, os.path.abspath(os.path.join(res_folder, f'AM_run_{b1}_{storage}_{steptime:03.1f}_{feed[0]}')))

    variables = {
        'b1_flowrate': str(b1*feed[0]),
        'storage_p': str(storage),
        }

    variables['t_ads'] = str(steptime)
    variables['feed_f'] = str(feed[0])
    variables['feed_y_fwd_co2'] = str(feed[1])
    variables['feed_y_fwd_N2'] = str(feed[2])
    variables['bed_mflow'] = str(feed[3])


    for v in variables:
        templ = templ.replace(v, variables[v])
    dest = os.path.join(root, 'Res', f'{run_name}',
              f'run_{b1}_{storage}_{steptime:03.1f}_{feed[0]}.ada')
    with open(dest, 'w') as o:
        o.write(templ)

    return res_folder, dest

def write_results_PPN6CH2TETA(Flowsheet, res_folder):
    """
    Fuction to extract and write the results of the simulation in text based
    format for the model PPN6CH2TETA.

    Parameters
    ----------
    Flowsheet : Acmsim.Application.Simulation.Flowsheet
        Flowsheet of the Aspen interface which allows access to all the
        relevant variables in the model.
    res_folder : str
        Folder in which the results are written.

    Returns
    -------
    None.

    """
    results = {}
    start_time = getattr(getattr(Flowsheet, 'FEED'), 'F').history.starttime
    interval = getattr(getattr(Flowsheet, 'FEED'), 'F').history.interval
    times = []
    for _ in getattr(getattr(Flowsheet, 'FEED'), 'F').history:
        times.append(start_time)
        start_time += interval

    for res in ['FEED', 'TAILPIPE', 'RECOUT', 'STORAGE']:
        results[res+'_F'] = [x for x in getattr(getattr(Flowsheet, res), 'F').history]
        results[res+'Y_Fwd("CO2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[0].history]
        results[res+'Y_Fwd("N2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[1].history]

    for res in ['feed_CO2', 'recout_CO2', 'tailpipe_CO2', 'storage_CO2']:
        results[res] = [x for x in getattr(Flowsheet, res).history]

    for res in ['BED1', 'BED2', 'BED3']:
        results[res+'_W(1, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[0].history]
        results[res+'_W(10, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[18].history]
        results[res+'_W(20, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[38].history]

        results[res+'_P(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[0].history]
        results[res+'_P(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[10].history]
        results[res+'_P(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[21].history]

        results[res+'_Ts(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[0].history]
        results[res+'_Ts(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[9].history]
        results[res+'_Ts(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[19].history]

    with open(os.path.join(res_folder, 'results.tsv'), 'w') as o:

        o.write('Time\t')
        o.write('\t'.join(sorted(list(results.keys()))))
        o.write('\n')

        for i in range(len(times)):
            o.write(str(times[i])+'\t')
            o.write('\t'.join([str(results[k][i]) for k in sorted(list(results.keys()))]))
            o.write('\n')




def write_results_zeolite13x(Flowsheet, res_folder):
    """
    Fuction to extract and write the results of the simulation in text based
    format for the model zeolite13x.

    Parameters
    ----------
    Flowsheet : Acmsim.Application.Simulation.Flowsheet
        Flowsheet of the Aspen interface which allows access to all the
        relevant variables in the model.
    res_folder : str
        Folder in which the results are written.

    Returns
    -------
    None.

    """
    #FEED: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #TAILPIPE: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #RECOUT: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #STORAGE: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #BED1-3.Layer[0]: W[0, 18, 38], P[0, 10, 21], Ts[0, 9, 19]

    results = {}
    start_time = getattr(getattr(Flowsheet, 'FEED'), 'F').history.starttime
    interval = getattr(getattr(Flowsheet, 'FEED'), 'F').history.interval
    times = []
    for _ in getattr(getattr(Flowsheet, 'FEED'), 'F').history:
        times.append(start_time)
        start_time += interval

    for res in ['FEED', 'TAILPIPE', 'PURGEOUTCO2', 'PURGEOUTH2O', 'STORAGECO2', 'STORAGEH2O', 'ROUTH2O', 'ROUTCO2']:
        results[res+'_F'] = [x for x in getattr(getattr(Flowsheet, res), 'F').history]
        results[res+'Y_Fwd("CO2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[0].history]
        results[res+'Y_Fwd("H2O")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[1].history]
        results[res+'Y_Fwd("N2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[2].history]

    for res in ['feed_CO2', 'routCO2_CO2', 'routH2O_CO2',  'tailpipe_CO2', 'storageCO2_CO2', 'storageH2O_CO2']:
        results[res] = [x for x in getattr(Flowsheet, res).history]

    for res in ['BED1', 'BED2']:
        results[res+'_W(1, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[0].history]
        results[res+'_W(10, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[18].history]
        results[res+'_W(20, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[38].history]

        results[res+'_W(1, H2O)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[1].history]
        results[res+'_W(10, H2O)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[19].history]
        results[res+'_W(20, H2O)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[39].history]

        results[res+'_P(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[0].history]
        results[res+'_P(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[10].history]
        results[res+'_P(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[21].history]

        results[res+'_Ts(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[0].history]
        results[res+'_Ts(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[9].history]
        results[res+'_Ts(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[19].history]

    with open(os.path.join(res_folder, 'results.tsv'), 'w') as o:

        o.write('Time\t')
        o.write('\t'.join(sorted(list(results.keys()))))
        o.write('\n')

        for i in range(len(times)):
            o.write(str(times[i])+'\t')
            o.write('\t'.join([str(results[k][i]) for k in sorted(list(results.keys()))]))
            o.write('\n')


def write_results_KAUST7(Flowsheet, res_folder):
    """
    Fuction to extract and write the results of the simulation in text based
    format for the model KAUST7.

    Parameters
    ----------
    Flowsheet : Acmsim.Application.Simulation.Flowsheet
        Flowsheet of the Aspen interface which allows access to all the
        relevant variables in the model.
    res_folder : str
        Folder in which the results are written.

    Returns
    -------
    None.

    """
    #FEED: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #TAILPIPE: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #RECOUT: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #STORAGE: F, Y_Fwd("CO2") [0], Y_Fwd("N2") [1]
    #BED1-3.Layer[0]: W[0, 18, 38], P[0, 10, 21], Ts[0, 9, 19]

    results = {}
    start_time = getattr(getattr(Flowsheet, 'FEED'), 'F').history.starttime
    interval = getattr(getattr(Flowsheet, 'FEED'), 'F').history.interval
    times = []
    for _ in getattr(getattr(Flowsheet, 'FEED'), 'F').history:
        times.append(start_time)
        start_time += interval

    for res in ['FEED', 'TAILPIPE', 'STORAGE']:

        results[res+'_F'] = [x for x in getattr(getattr(Flowsheet, res), 'F').history]
        results[res+'Y_Fwd("CO2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[0].history]
        results[res+'Y_Fwd("N2")'] = [x for x in getattr(getattr(Flowsheet, res), 'Y_fwd')[1].history]

    for res in ['feed_CO2', 'tailpipe_CO2', 'storage_CO2']:
        results[res] = [x for x in getattr(Flowsheet, res).history]

    for res in ['BED']:
        results[res+'_W(1, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[0].history]
        results[res+'_W(10, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[18].history]
        results[res+'_W(20, CO2)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'W')[38].history]

        results[res+'_P(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[0].history]
        results[res+'_P(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[10].history]
        results[res+'_P(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'P')[21].history]

        results[res+'_Ts(0)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[0].history]
        results[res+'_Ts(10)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[9].history]
        results[res+'_Ts(21)'] = [x for x in getattr(getattr(getattr(Flowsheet, res), 'Layer')[0], 'Ts')[19].history]

    with open(os.path.join(res_folder, 'results.tsv'), 'w') as o:

        o.write('Time\t')
        o.write('\t'.join(sorted(list(results.keys()))))
        o.write('\n')

        for i in range(len(times)):
            o.write(str(times[i])+'\t')
            o.write('\t'.join([str(results[k][i]) for k in sorted(list(results.keys()))]))
            o.write('\n')

def run_simulation(res_folder, file_path, write_results):
    """
    This function uses the data present in the created folders and the
    formatted model and it opens Aspen and runs a simulation with the
    parameters specified in the ada file.


    Parameters
    ----------
    res_folder : str
        Folder in which the results are written.
    file_path : str
        Location of the ada file to run.
    write_results : function
        Function to read the results from Aspen and write the results in text
        based format.

    Returns
    -------
    proc : subprocess.Process
        Process in which Aspen is spawned and the simulation run.
    logs_failed : str
        Aspen log stream which is parsed to check for errors.

    """
    proc = subprocess.Popen([asp_path, file_path])
    time.sleep(20)

    while True:
        try:
            ACMapp = client.GetActiveObject("ACM application")
            break
        except:
            time.sleep(20)

    Acmsim = ACMapp.Simulation
    logs_failed = []
    try:
        Acmsim.Run(True)
    except:
        logs_failed = list(Acmsim.Application.Simulation.OutputLogger.messages)

    Acmsim.Application.Simulation.Results.Refresh()
    time.sleep(10)
    write_results(Acmsim.Application.Simulation.Flowsheet, res_folder)
    with open(os.path.join(res_folder, 'log.txt'), 'a') as log:
        logs = logs_failed + list(Acmsim.Application.Simulation.OutputLogger.messages)
        log.write('\n'.join(logs))

    proc.kill()
    proc = None
    if any(['Check your model equations and variable bounds' in l for l in logs]):
        proc = True
    return proc, logs_failed

def main(*args, create_files_and_folders=create_files_and_folders_PPN6CH2TETA, write_results=write_results_PPN6CH2TETA):
    """
    Main function that wraps all the other functions defined. It also checks
    for the Success or Failure of the simulation (eg. if the simulation does
    not converge).

    Parameters
    ----------
    *args : *float
        The values of the variables that have to change in each model.
    create_files_and_folders : function, optional
        The function to prepare the simulation folders and files.
                        The default is create_files_and_folders_PPN6CH2TETA.
    write_results : function, optional
        The function to write the results of the simulation.
                        The default is write_results_PPN6CH2TETA.

    Returns
    -------
    None.

    """
    fil_fol_out = create_files_and_folders(*args)

    if not fil_fol_out:
        return
    res_folder, dest = fil_fol_out
    exit_value, logs_failed = run_simulation(res_folder, dest, write_results)

    if exit_value:

        print('FAILED')

        try:
            open(os.path.join(res_folder, 'FAIL'), 'w').close()
            time.sleep(10)
            os.rename(res_folder, res_folder.replace(basename, 'F_'+basename))
        except:
            pass
        return

    else:
        print('SUCCEDED')
        try:
            open(os.path.join(res_folder, 'S'), 'w').close()
            time.sleep(10)
            os.rename(res_folder, res_folder.replace(basename, 'S_'+basename))
        except:
            pass

def PPN6CH2TETA():
    """
    Function to perform the evaluation of the model PPN6CH2TETA in different
    conditions.

    Returns
    -------
    None.

    """
    recout_p = [0.9, 0.8, 0.5, 0.1, 0.01]
    storage_p = [0.9, 0.8, 0.5, 0.1, 0.01]
    co_data2_steptime = [500.0, 450.0, 300.0, 150.0, 100.0, 50.0]
    #           F kmol/s    yco2    yn2     exhaust kg/s
    feed_f = [[3.617*1e-3, 0.138, 1-0.138, 0.1096/3],    #90kph
            [4.138*1e-3, 0.126, 1-0.126, 0.12486/3]]   #105kph

    tot = len(recout_p)*len(storage_p)*len(co_data2_steptime)*len(feed_f)
    i = 0
    for rec in recout_p:
        for sto in storage_p:
            for co in co_data2_steptime:
                for fe in feed_f:
                    i += 1
                    print(f'         Running simulation for {rec}, {sto}, {fe[0]}, {co:03.1f}\t{i}/{tot}', end='\r')
                    main(rec, sto, fe, co, create_files_and_folders=create_files_and_folders_PPN6CH2TETA, write_results=write_results_PPN6CH2TETA)
def KAUST7():
    """
    Function to perform the evaluation of the model KAUST7 in different
    conditions.

    Returns
    -------
    None.

    """
    storage_p = [0.9, 0.5, 0.1, 0.05, 0.02, 0.01]
    #           F kmol/s    yco2    yn2     exhaust kg/s
    feed_f = [[3.617*1e-3/2, 0.138, 1-0.138, 0.1096/2],    #90kph
            [4.138*1e-3/2, 0.126, 1-0.126, 0.12486/2]]   #105kph
    t_ads = [200, 150, 100, 50, 30]
    b1_flowrate = [0.0, 0.05, 0.01, 0.005]

    tot = len(storage_p)*len(t_ads)*len(feed_f)*len(b1_flowrate)
    i = 0
    for sto in storage_p:
        for co in t_ads:
            for b1 in b1_flowrate:
                for fe in feed_f:
                    i += 1
                    print(f'         Running simulation for {b1}, {sto}, {fe[0]}, {co:03.1f}\t{i}/{tot}', end='\r')
                    main(b1, sto, fe, co, create_files_and_folders=create_files_and_folders_KAUST7, write_results=write_results_KAUST7)
def zeolite13x():
    """
    Function to perform the evaluation of the model zeolite13x in different
    conditions.

    Returns
    -------
    None.

    """
    storageh2o_p = [0.9, 0.5, 0.1, 0.01]
    storageco2_p = [0.9, 0.5, 0.1, 0.01]
    vspurgeh2o = [0.02, 0.01, 0.005]

    t_ads = [150.0, 100.0, 50.0, 30.0]
    #           F kmol/s    yco2    yh2o        yn2             exhaust kg/s
    feed_f = [[3.748*1e-3, 0.13278, 0.03497, 1-0.13278-0.03497, 0.1096],    #90kph
            [4.296*1e-3, 0.13912, 0.03497, 1-0.13912-0.03497, 0.12486]]   #105kph

    tot = len(storageh2o_p)*len(storageco2_p)*len(t_ads)*len(feed_f)*len(vspurgeh2o)
    i = 0
    for stoh in storageh2o_p:
        for stoc in storageco2_p:
            for co in t_ads:
                for vph in vspurgeh2o:
                    for fe in feed_f:
                        i += 1
                        print(f'         Running simulation for {stoh}, {stoc}, {vph}, {fe[0]}, {co:03.1f}\t{i}/{tot}', end='\r')
                        main(stoh, stoc, vph, fe, co, create_files_and_folders=create_files_and_folders_zeolite13x, write_results=write_results_zeolite13x)

if __name__ == '__main__':

    #The config file is expected to contain two key value pairs tab separated
    #One value aspen_path should specify the path to AspenModeler.exe.
    #One value project_path should specify the path to the root directory in
    #which the simulations have to be performed.

    config = {}
    with open('config.cfg') as cfg:
        for line in cfg.readlines():
            line = line.strip().split('\t')
            config[line[0]] = line[1]

    asp_path = os.path.abspath(config['aspen_path'])

    projectPath = os.path.abspath(filedialog.askdirectory(initialdir=config['project_path']))

    if not projectPath:
        print('No project selected')
        sys.exit()

    template_path = os.path.abspath(os.path.join(projectPath, 'template'))
    properties_folder = template_path
    for t in os.listdir(template_path):
        if '_TEMPLATE.ada' in t:
            template_path = os.path.abspath(os.path.join(template_path, t))
            basename = os.path.basename(template_path).replace('_TEMPLATE.ada', '')
        if os.path.isdir(os.path.abspath(os.path.join(template_path, t))) and t.startswith('AM_'):
            properties_folder = os.path.abspath(os.path.join(template_path, t))

    root = projectPath
    if 'PPN' in projectPath:
        PPN6CH2TETA()
    elif 'KAUST' in projectPath:
        KAUST7()
    elif 'Zeolite' in projectPath:
        zeolite13x()