"""
This is a simple example to show how to set up and simulation and utilize some of emspy's features.
This implements simple rule-based thermostat control based on the time of day, for a single zone of a 5-zone office
building. Other data is tracked and reported just for example.

The same functionality implemented in this script could be done much more simply, but I wanted to provide some exposure
to the more complex features that are really useful when working with more complicated RL control tasks; Such as the use
of the MdpManager to handle all of the simulation data and EMS variables.
"""
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import math

from eppy import modeleditor
from eppy.modeleditor import IDF

import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
from model_RNN import Linear_QNet, QTrainer
from agent_RNN_Load_Eval import Agent
from helper_plot import plot

from emspy import EmsPy, BcaEnv, MdpManager

"""
Folder Structure for multiple records
	building-insubase-massup
	building-insuup-massbase
	buil... x9 (rows in final table)
		in.idf
		BaselineEP
		RL24hAllRNN
		RL24h... x6 (columns in final table)
			eplusout.csv
			agent loss
			agent memory
			eplus.htm
			...
					Post Processing
					Energy use total Joules
					% time too cold
					% time too hot
"""

base_path = r"W:\Insync\GDrive\Main\TU Delft\Thesis\DRL runs 11"
model_base_path = r"W:\Insync\GDrive\Main\TU Delft\Thesis\DRL runs 09 models"

# -- FILE PATHS --
# * E+ Download Path *
ep_path = "C:/EnergyPlusV22-2-0/"  # path to E+ on system / or use V22-2-0/

# Weather Path
ep_weather_path = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\GBR_WAL_Lake.Vyrnwy.034100_TMYx.2007-2021.epw"  # EPW weather file
# 7 years data
# ep_weather_path = r"W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Weather data\combined weather data\GBR_WAL_Combined 7 years.epw"  # EPW weather file


# name_of_control_this_run = 'RLBaseNoForesight' 
list_names = ['RL24hAllRNN', 'RL04hAllRNN', 'RLBaseNoForesight', 'RL24hNoSolarRNN', 'RL04hNoSolarRNN', 'RL04hFlatInput']

for name_of_control_this_run in list_names:

    # 9 buliding types that will be cycled through
    building_types_list = ['Building-InsuBASE-MassBASE',
                            'Building-InsuBASE-MassDW',
                            'Building-InsuBASE-MassUP',
                            'Building-InsuDW-MassBASE',
                            'Building-InsuDW-MassDW',
                            'Building-InsuDW-MassUP',
                            'Building-InsuUP-MassBASE',
                            'Building-InsuUP-MassDW',
                            'Building-InsuUP-MassUP']

    # loop to go through each building per specified technique
    for builidng_enum_no, unique_building_name in enumerate(building_types_list):



        # print(f'Control foresight is {control_foresight} hours')
        print(f'Control type is {name_of_control_this_run}')
        # print(f'Solar is set to {solar_included} ')
        print(f'Building is {unique_building_name} ')

        # Edit idf RunPeriod to 3 years also for baseline, only 1 year
        iddfile = r"C:\EnergyPlusV22-2-0\Energy+.idd"
        fname1 = os.path.join(base_path, unique_building_name, 'in.idf')
        IDF.setiddname(iddfile)
        idf1 = IDF(fname1)
        idf1.idfobjects['RUNPERIOD'][0].End_Month = 12
        idf1.idfobjects['RUNPERIOD'][0].End_Day_of_Month = 31
        # if name_of_control_this_run == 'EPBaseline': # 1 year only
        idf1.idfobjects['RUNPERIOD'][0].End_Year = idf1.idfobjects['RUNPERIOD'][0].Begin_Year
        # else: # 3 years run period
        #     idf1.idfobjects['RUNPERIOD'][0].End_Year = idf1.idfobjects['RUNPERIOD'][0].Begin_Year + 2
        idf1.newidfobject('Output:Variable')
        idf1.idfobjects['Output:Variable'][-1].Variable_Name = 'Surface Heat Storage Rate per Area'
        idf1.newidfobject('Output:Variable')
        idf1.idfobjects['Output:Variable'][-1].Variable_Name = 'Surface Heat Storage Energy'
        
        idf1.saveas(os.path.join(base_path, unique_building_name, 'in_edit.idf'))



        # model_path_this_run = os.path.join(model_base_path, f'{name_of_control_this_run}-{unique_building_name}')
        model_path_this_run = os.path.join(model_base_path, f'{name_of_control_this_run}-Building-InsuBASE-MassBASE')

        current_directory = os.path.join(base_path, unique_building_name, name_of_control_this_run)
        
        if not os.path.exists(current_directory):
            os.makedirs(current_directory)

        os.chdir(current_directory) # EP /out folder will be saved to this location, also model loss and memory csv

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # a workaround to an error I encountered when running sim  
        # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

    
        # IDF File / Modification Paths
        # idf_file_name = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\Base_model simple schedule.idf"  # building energy model (BEM) IDF file
        idf_file_name = os.path.join(base_path, unique_building_name, 'in_edit.idf')

        # Output .csv Path (optional)
        # cvs_output_path = r'C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\dataframes_output_test.csv'



        # STATE SPACE (& Auxiliary Simulation Data)
        """
        See mdpmanager.MdpManager.generate_mdp_from_tc() to understand this structure for initializing and managing variables, 
        and creating optional encoding functions/args for automatic encoding/normalization on variables values.

        Note, however, MdpManager is not required. You can choose to manage variables on your own.
        """

        # --- create EMS Table of Contents (TC) for sensors/actuators ---
        # int_vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
        # vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
        # meters_tc = {"attr_handle_name": [("meter_name")],...}
        # actuators_tc = {"attr_handle_name": [("component_type", "control_type", "actuator_key")],...}
        # weather_tc = {"attr_name": [("weather_metric"),...}

        zn1 = 'Z1_Ground_Floor' # hint: search .idf for 'spacelist,' or 'zonelist,' this shows the spaces/zones
        zn2 = 'Z2_First_Floor'

        tc_intvars = {}

        tc_vars = { # These must be present in .idf file and match possible values in .rdd file.
                    # example in .idf: Output:Variable,*,Schedule Value,Timestep;
            # Building - same as in .rdd file. rdd file is the variable dictionary.
            # 'hvac_operation_sched': [('Schedule Value', ' SeasonalSchedule_7de19e4d')],  # is building 'open'/'close'? # hint: search .idf for 'schedule:year,' and it will show the name of the schedule, given there is one primary schedule.
            # 'people_occupant_count': [('People Occupant Count', zn1)],  # number of people per zn1
            # -- Zone 0 (Core_Zn) --
            'zn1_temp': [('Zone Operative Temperature', zn1)],  # deg C
            'zn2_temp': [('Zone Operative Temperature', zn2)],  # deg C
            'zn1_Airtemp': [('Zone Mean Air Temperature', zn1)],  # deg C
            'zn2_Airtemp': [('Zone Mean Air Temperature', zn2)],  # deg C
            'zn1_Radtemp': [('Zone Mean Radiant Temperature', zn1)],  # deg C
            'zn2_Radtemp': [('Zone Mean Radiant Temperature', zn2)],  # deg C
            'zn1_RH': [('Zone Air Relative Humidity', zn1)],  # %RH
            'zn2_RH': [('Zone Air Relative Humidity', zn2)],  # %RH
            'zn1_IntWallMassRate': [('Surface Heat Storage Rate per Area', 'INT_WALLS_MASS_GROUND::Z1_GROUND_FLOOR_SPACE')],  # W/m2
            'zn1_StairsMassRate': [('Surface Heat Storage Rate per Area', 'STAIRS_MASS::Z1_GROUND_FLOOR_SPACE')],  # W/m2
            'zn2_IntWallMassRate': [('Surface Heat Storage Rate per Area', 'INT_WALLS_MASS_FIRST::Z2_FIRST_FLOOR_SPACE')],  # W/m2
            'zn1_IntWallMassEnergy': [('Surface Heat Storage Energy', 'INT_WALLS_MASS_GROUND::Z1_GROUND_FLOOR_SPACE')],  # J
            'zn1_StairsMassEnergy': [('Surface Heat Storage Energy', 'STAIRS_MASS::Z1_GROUND_FLOOR_SPACE')],  # J
            'zn2_IntWallMassEnergy': [('Surface Heat Storage Energy', 'INT_WALLS_MASS_FIRST::Z2_FIRST_FLOOR_SPACE')],  # J
        }

        """
        NOTE: meters currently do not accumulate their values within there sampling interval during runtime, this happens at the
        end of the simulation as a post-processing step. These will behave a lot like just a collection of EMS variables. See
        UnmetHours for more info.
        """
        tc_meters = {
            # Building-wide
            'electricity_facility': ['Electricity:Facility'],  # J
            # 'electricity_HVAC': [('Electricity:HVAC')],  # J
            'electricity_heating': [('DistrictHeating:Facility')],  # J
            # 'electricity_cooling': [('Cooling:Electricity')],  # J
            # 'gas_heating': [('NaturalGas:HVAC')]  # J
        }
        # detailed weather which is not for current time cannot be called here. Instead I use the self.bca.get_weather_forecast inside the Agent
        tc_weather = {
            'oa_rh': [('outdoor_relative_humidity')],  # %RH
            'oa_db': [('outdoor_dry_bulb')],  # deg C
            'oa_pa': [('outdoor_barometric_pressure')],  # Pa
            'sun_up': [('sun_is_up')],  # T/F
            'rain': [('is_raining')],  # T/F
            'snow': [('is_snowing')],  # T/F
            'wind_dir': [('wind_direction')],  # deg
            'wind_speed': [('wind_speed')],  # m/s
            'beam_solar': [('beam_solar')], # Wh/m2
            'diffuse_solar': [('diffuse_solar')] # Wh/m2
            # 'wind_speed_tomorrow': [('wind_speed'), 'tomorrow', 12, 1]  # m/s
        } # 'beam_solar', 'diffuse_solar' # Wh/m2

        # ACTION SPACE
        """
        NOTE: only zn1 (CoreZn) has been setup in the model to allow 24/7 HVAC setpoint control. Other zones have default
        HVAC operational schedules and night cycle managers that prevent EMS Actuator control 24/7. Essentially, at times the 
        HVAV is "off" and can't be operated. If all zones are to be controlled 24/7, they must be implemented as CoreZn.
        See the "HVAC Systems" tab in OpenStudio to zone configurations.
        """
        tc_actuators = { # 'user_var_name': ['component_type', 'control_type', 'actuator_key'] within the dict
            # HVAC Control Setpoints
            # 'zn1_cooling_sp': [('Zone Temperature Control', 'Cooling Setpoint', zn1)],  # deg C
            'zn1_heating_sp': [('Zone Temperature Control', 'Heating Setpoint', zn1)],  # deg C
            'zn2_heating_sp': [('Zone Temperature Control', 'Heating Setpoint', zn2)],  # deg C
        } # Z2_LOUVRE_20,AirFlow Network Window/Door Opening,Venting Opening Factor,[Fraction]
        # opening window from .edd APERTURE_E02C3A1A_OPENING,Zone Ventilation,Air Exchange Flow Rate,[m3/s]
        # Window actuators
        z1_louvre_names_list = [f'z1_louvre_{x+1}' for x in range(20)]
        z2_louvre_names_list = [f'z2_louvre_{x+1}' for x in range(38)]
        for i, each in enumerate(z1_louvre_names_list):
            tc_actuators[f'z1_louvre_act_{i+1}'] = [('AirFlow Network Window/Door Opening', 'Venting Opening Factor', each)]
        for i, each in enumerate(z2_louvre_names_list):
            tc_actuators[f'z2_louvre_act_{i+1}'] = [('AirFlow Network Window/Door Opening', 'Venting Opening Factor', each)]

        all_zones_louvre_act_names_list = [f'z1_louvre_act_{x+1}' for x in range(20)] + [f'z2_louvre_act_{x+1}' for x in range(38)]



        # -- INSTANTIATE 'MDP' --
        my_mdp = MdpManager.generate_mdp_from_tc(tc_intvars, tc_vars, tc_meters, tc_weather, tc_actuators)

        # -- Simulation Params --
        calling_point_for_callback_fxns = EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop during simulation
        sim_timesteps = 1  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

        # -- Create Building Energy Simulation Instance --
        sim = BcaEnv(
            ep_path=ep_path,
            ep_idf_to_run=idf_file_name,
            timesteps=sim_timesteps,
            tc_vars=my_mdp.tc_var,
            tc_intvars=my_mdp.tc_intvar,
            tc_meters=my_mdp.tc_meter,
            tc_actuator=my_mdp.tc_actuator,
            tc_weather=my_mdp.tc_weather
        )


        # Create agent instance
        my_agent = Agent(sim, my_mdp, all_zones_louvre_act_names_list, name_of_control_this_run, model_path_this_run)

        # Set your callback function (observation and/or actuation) function for a given calling point
        sim.set_calling_point_and_callback_function(
            calling_point=calling_point_for_callback_fxns,
            observation_function=my_agent.observation_function,  # optional
            actuation_function=my_agent.actuation_function,  # optional
            update_state=True,
            update_observation_frequency=1,
            update_actuation_frequency=1
        )

        # -- RUN BUILDING SIMULATION --
        sim.run_env(ep_weather_path)
        sim.reset_state()  # reset when done


        # my_agent.model.save(f'{name_of_control_this_run}-{unique_building_name}')


        # save agent memory
        # current_directory = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work"
        # current_directory is the path for the buliding and the run name, specified at top
        path_record_name = os.path.join(current_directory, 'agent memory.csv')
        df = pd.DataFrame(my_agent.memory, columns= ["state_prev", "action", "next_reward", "next_state", "game_over"])
        df.to_csv(path_or_buf = path_record_name)


        # path_record_loss_list = os.path.join(current_directory, 'agent loss.csv')
        # loss_list = my_agent.trainer.loss_record
        # dfloss = pd.DataFrame(loss_list)
        # dfloss.to_csv(path_or_buf = path_record_loss_list)


        # Manual dataframe
        path_record_manual_dataframe = os.path.join(current_directory, 'manual dataframe.csv')
        manual_df = pd.DataFrame(my_agent.manual_dataframe)
        manual_df.to_csv(path_or_buf = path_record_manual_dataframe)

        # -- Sample Output Data --
        # output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too


        # -- Plot Results --
        # fig, ax = plt.subplots()
        # output_dfs['var'].plot(y='zn1_temp', use_index=True, ax=ax)
        # output_dfs['weather'].plot(y='oa_db', use_index=True, ax=ax)
        # output_dfs['meter'].plot(y='electricity_HVAC', use_index=True, ax=ax, secondary_y=True)
        # output_dfs['actuator'].plot(y='zn1_heating_sp', use_index=True, ax=ax)
        # output_dfs['actuator'].plot(y='zn1_cooling_sp', use_index=True, ax=ax)
        # plt.title('zn1 Temps and Thermostat Setpoint for Year')

        # Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes


