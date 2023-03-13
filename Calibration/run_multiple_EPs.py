import os
import pandas as pd
from path import Path

import eppy
from eppy import modeleditor
from eppy.modeleditor import IDF
import eppy.json_functions as json_functions
from eppy.results import readhtml

from energyplus_wrapper import EPlusRunner, ensure_eplus_root
import joblib
from joblib import Parallel, delayed

# x take in pandas of generation
# x use eppy to load and edit base_idf file to fit children
# use run_many_EP from EP-wrapper to run whole generation and save to subfolders
# x use eppy to read html results - https://eppy.readthedocs.io/en/latest/Outputs_Tutorial.html
# return html single kwh/m2 converted from MJ/m2 to function

def run_EPs_parallel_custom(current_gen):
    # df_all_cols = pd.read_csv(r"C:\Users\sebas\Documents\GitHub\ClimAIte\data_records\record_0", sep=',')
    df_all_cols = current_gen

    fname1_base_idf = r"W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Calibration files\Base_model.idf"
    epw_main = r"W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Calibration files\GBR_WAL_Lake.Vyrnwy.034100_TMYx.2007-2021.epw"

    eplus_root = "C:\EnergyPlusV22-2-0" #for ep-wrapper
    archive_folder = Path("W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Calibration files\IDFs\IDF results backups").abspath() # for ep-wrapper file back-up

    iddfile = r"C:\EnergyPlusV22-2-0\Energy+.idd"
    IDF.setiddname(iddfile)
    idf_base = IDF(fname1_base_idf)

    # take dataframe, duplicate HVAC column for Zone 2. Then take only idf columns and get index start.

    df_all_cols['idf.ZoneHVAC:IdealLoadsAirSystem.Z2_First_Floor_9e34b4d3 Ideal Loads Air System.Sensible_Heat_Recovery_Effectiveness'] = df_all_cols['idf.ZoneHVAC:IdealLoadsAirSystem.Z1_Ground_Floor_48970ba6 Ideal Loads Air System.Sensible_Heat_Recovery_Effectiveness']
    df_idf_only = df_all_cols.filter(regex='idf', axis=1)
    df_idf_only.head()
    list_of_strings = []
    keys = df_idf_only.columns.values.tolist()
    index_start = df_idf_only.index.values.tolist()[0]
    for row in range(df_idf_only.shape[0]):
            dict = {}
            for i in keys:
                    dict[i] = df_idf_only.at[index_start + row, i]
            list_of_strings.append(dict)

    # create all IDFs

    unique_name_list = []

    for i in range(df_idf_only.shape[0]):
        current_name = df_all_cols.at[index_start + i, 'Unique Name']
        unique_name_list.append(current_name)
        json_str = list_of_strings[i]
        json_functions.updateidf(idf_base, json_str)
        idf_base.saveas(f'W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Calibration files\IDFs\Gene-{current_name}.idf')



    # prep files for parallel
    samples = {}
    for name in unique_name_list:
        samples[name] = f'W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Calibration files\IDFs\Gene-{name}.idf', epw_main

    # run in parallel
    runner = EPlusRunner(eplus_root)

    with joblib.parallel_backend("loky", n_jobs=10):
        sims = runner.run_many(samples, backup_strategy='always', backup_dir=archive_folder, version_mismatch_action='ignore')

    list_of_sims = list(sims.values())

    # Results from sims dictionaries
    EUI_results = []
    for simulation in list_of_sims:
        assert simulation.reports['Annual_Building_Utility_Performance_Summary_for_Entire_Facility']['Site and Source Energy'].columns[1] == 'Energy Per Total Building Area [MJ/m2]'
        MJ_m2 = simulation.reports['Annual_Building_Utility_Performance_Summary_for_Entire_Facility']['Site and Source Energy'].iat[0,1]
        EUI_kWh_m2 = MJ_m2 * 0.2777777 #conversion from MJ to kWh
        EUI_results.append(EUI_kWh_m2)

    return EUI_results