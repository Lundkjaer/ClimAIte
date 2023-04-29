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

base_path = r"W:\Insync\GDrive\Main\TU Delft\Thesis\DRL runs 07"

# name_of_control_this_run = 'RLBaseNoForesight' 
list_names = ['RL24hAllRNN', 'RL04hAllRNN', 'RLBaseNoForesight']#, 'RL24hNoSolarRNN', 'RL04hNoSolarRNN', 'RL04hFlatInput']

for name_of_control_this_run in list_names:

    if name_of_control_this_run[2:4] == '24':
        control_foresight = 24
    elif name_of_control_this_run[2:4] == '04':
        control_foresight = 4
    else:
        control_foresight = 0

    if 'NoSolar' in name_of_control_this_run:
        solar_included = False
    else:
        solar_included = True

    dict_Qnet_flat_input_size = {
                'EPBaseline':0, #  it's 0, network redundant
                'RLBaseNoForesight':26,
                'RL24hAllRNN':26,
                'RL24hNoSolarRNN':26,
                'RL04hAllRNN':26,
                'RL04hNoSolarRNN':26,
                'RL04hFlatInput':26} 
    Qnet_flat_input_size = dict_Qnet_flat_input_size[name_of_control_this_run]

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

        print(f'Control foresight is {control_foresight} hours')
        print(f'Control type is {name_of_control_this_run}')
        print(f'Solar is set to {solar_included} ')
        print(f'Building is {unique_building_name} ')

        # Edit idf RunPeriod to 3 years also for baseline, only 1 year
        iddfile = r"C:\EnergyPlusV22-2-0\Energy+.idd"
        fname1 = os.path.join(base_path, unique_building_name, 'in.idf')
        IDF.setiddname(iddfile)
        idf1 = IDF(fname1)
        idf1.idfobjects['RUNPERIOD'][0].End_Month = 12
        idf1.idfobjects['RUNPERIOD'][0].End_Day_of_Month = 31
        if name_of_control_this_run == 'EPBaseline': # 1 year only
            idf1.idfobjects['RUNPERIOD'][0].End_Year = idf1.idfobjects['RUNPERIOD'][0].Begin_Year
        else: # 3 years run period
            idf1.idfobjects['RUNPERIOD'][0].End_Year = idf1.idfobjects['RUNPERIOD'][0].Begin_Year + 2
        idf1.newidfobject('Output:Variable')
        idf1.idfobjects['Output:Variable'][-1].Variable_Name = 'Surface Heat Storage Rate per Area'
        idf1.newidfobject('Output:Variable')
        idf1.idfobjects['Output:Variable'][-1].Variable_Name = 'Surface Heat Storage Energy'
        
        idf1.saveas(os.path.join(base_path, unique_building_name, 'in_edit.idf'))


        current_directory = os.path.join(base_path, unique_building_name, name_of_control_this_run)
        
        if not os.path.exists(current_directory):
            os.makedirs(current_directory)

        os.chdir(current_directory) # EP /out folder will be saved to this location, also model loss and memory csv



        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # a workaround to an error I encountered when running sim  
        # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

        # -- FILE PATHS --
        # * E+ Download Path *
        ep_path = "C:/EnergyPlusV22-2-0/"  # path to E+ on system / or use V22-2-0/


        # IDF File / Modification Paths
        # idf_file_name = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\Base_model simple schedule.idf"  # building energy model (BEM) IDF file
        idf_file_name = os.path.join(base_path, unique_building_name, 'in_edit.idf')

        # Weather Path
        ep_weather_path = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\GBR_WAL_Lake.Vyrnwy.034100_TMYx.2007-2021.epw"  # EPW weather file
        # 7 years data
        # ep_weather_path = r"W:\Insync\GDrive\Main\TU Delft\Thesis\EnergyPlus\Weather data\combined weather data\GBR_WAL_Combined 7 years.epw"  # EPW weather file

        # Output .csv Path (optional)
        # cvs_output_path = r'C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\dataframes_output_test.csv'


        def normalise(xvalue, xmin, xmax):
            xvalue = np.clip(xvalue, xmin, xmax)
            # return round( (xvalue - xmin) / (xmax - xmin) , 5) # [0,1]
            return round( 2* ((xvalue - xmin) / (xmax - xmin)) -1 , 5) # [-1,1]

        def denormalise(xvalue_norm, xmin, xmax):
            # return round( xvalue_norm * (xmax - xmin) + xmin , 2)
            return round( (xvalue_norm + 1) * (xmax - xmin) *0.5 + xmin , 2)


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

        MAX_MEMORY = 100_000 # roughly 11 years of hourly data
        BATCH_SIZE = 1000
        LR = 0.0001
        manual_dataframe = []

        class Agent:
            """
            Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
            scope throughout the simulation.
            Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
            (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
            instance.

            NOTE: The observation and actuation functions are callback functions that, depending on your configuration (calling
            points,

            Depending on your implementation, an observation function may not be needed. The observation function just
            allows for extra flexibility in gathering state data from the simulation, as it happens before the actuation
            function is calle, and it can be used to return and store "reward" throughout the simulation, if desired.

            The actuation function is used to take actions. And must return an "actuation dictionary" that has the specific
            EMS actuator variable with its corresponding control value.
            """
            def __init__(self, bca: BcaEnv, mdp: MdpManager):
                self.bca = bca
                self.mdp = mdp

                # simplify naming of all MDP elements/types
                self.vars = mdp.ems_type_dict['var']  # all MdpElements of EMS type var
                self.meters = mdp.ems_type_dict['meter']  # all MdpElements of EMS type meter
                self.weather = mdp.ems_type_dict['weather']  # all MdpElements of EMS type weather
                self.actuators = mdp.ems_type_dict['actuator']  # all MdpElements of EMS type actuator

                # get just the names of EMS variables to use with other functions
                self.var_names = mdp.get_ems_names(self.vars)
                self.meter_names = mdp.get_ems_names(self.meters)
                self.weather_names = mdp.get_ems_names(self.weather)
                self.actuator_names = mdp.get_ems_names(self.actuators)

                # simulation data state
                self.zn1_temp = None  # deg C
                self.zn2_temp = None  # deg C
                self.time = None
                self.day_of_week = None
                    # first element next hour, reverse list for RNN
                self.future_work_hour_booleans = [] # school open? boolean
                self.future_work_hour_booleans_len = 72
                self.future_global_rad = [] # Global radiation Wh / m2, 0-1000
                self.future_global_rad_len = 24
                self.future_diffuse_rad = [] # diffuse radiation Wh / m2, 0-1000
                self.future_diffuse_rad_len = 24
                self.future_ext_temp = [] # external temperatures future
                self.future_ext_temp_len = 24

                # self.rl_input_params = [self.zn1_temp, # the no. items must match the QNet input size
                #                         self.time.hour,
                #                         self.day_of_week]

                self.work_hours_heating_setpoint = 20  # deg C
                self.work_hours_cooling_setpoint = 25  # deg C
                self.off_hours_heating_setpoint = 15  # deg C # not currently used
                self.off_hours_cooilng_setpoint = 30  # deg C # not currently used
                self.work_day_start = datetime.time(8, 0)  # day starts 6 am
                self.work_day_end = datetime.time(16, 0)  # day ends at 8 pm

                # print reporting
                self.print_every_x_hours = 6

                # DRL

                self.n_game_steps = 0
                self.epsilon = 0 # randomness, greedy/exploration. This is overridden in def action()
                self.gamma = 0.9 # discount rate
                self.memory = deque(maxlen=MAX_MEMORY) # if memory larger, it calls popleft()
                self.model = Linear_QNet(Qnet_flat_input_size,300,300,11, control_foresight, solar_included) # neural network (input, hidden x5, output) # nnSizeOBS
                self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

            
            def get_state(self, game): #observation in BcaEnv
                pass

            def remember(self, state, action, reward, next_state, game_over):
                self.memory.append((state, action, reward, next_state, game_over)) #deque will popleft() if max_memory is reached. Extra () parantheses to store values as single tuple

            def train_long_memory(self):
                # print('Start train long memory loop')
                if len(self.memory) > BATCH_SIZE:
                    mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples
                else:
                    mini_sample = self.memory
                    
                states, actions, rewards, next_states, game_overs = zip(*mini_sample) # unpack into lists rather than combined tuples
                self.trainer.train_step(states, actions, rewards, next_states, game_overs)
                # print('Finished training long memory loop')

            def train_short_memory(self, state, action, reward, next_state, game_over):
                self.trainer.train_step(state, action, reward, next_state, game_over)

            def reward_calculation(self): 
                total_reward = 0
                # done - make temp scale piecewise and increasing
                # get meter reading for prev kwh spending
                # can use electricity_facility? - this seems to be instantenous
                # total_reward -= prev_kwh * 2
                total_reward -= round(self.bca.get_ems_data(['electricity_heating']) / 6_000_000 / 10, 4)

                if self.work_day_start <= self.time.time() < self.work_day_end:  #
                    # during workday
                    if min(self.zn1_temp, self.zn2_temp) < self.work_hours_heating_setpoint:
                        total_reward -= 10 * ( self.work_hours_heating_setpoint - min(self.zn1_temp, self.zn2_temp) )

                    if max(self.zn1_temp, self.zn2_temp) > self.work_hours_cooling_setpoint:
                        total_reward -= 10 * - ( self.work_hours_cooling_setpoint - max(self.zn1_temp, self.zn2_temp) )
                #     heating_setpoint = work_hours_heating_setpoint
                #     cooling_setpoint = work_hours_cooling_setpoint
                #     thermostat_settings = 'Work-Hours Thermostat'
                # else:
                #     # off work
                #     heating_setpoint = off_hours_heating_setpoint
                #     cooling_setpoint = off_hours_cooilng_setpoint
                #     thermostat_settings = 'Off-Hours Thermostat'

                return total_reward


            def get_action(self, state): #action/actuation in BcaEnv
                # random moves: exploration / exploitation
                self.epsilon = 30_000 - self.n_game_steps
                final_move = [0 for x in range(11)] # [0,0,0] in snake game # nnSizeOBS
                if random.randint(0,30_000) < self.epsilon or random.randint(0,100) < 5:
                    move = random.randint(0,10) # from snake with argmax #nnSizeOBS
                    final_move[move] = 1
                else:
                    state0 = torch.tensor(state, dtype=torch.float)
                    prediction = self.model(state0)
                    move = torch.argmax(prediction).item() # picks output with highest predicted reward
                    final_move[move] = 1 # from snake with argmax

                
                return final_move

            def observation_function(self):
                # -- FETCH/UPDATE SIMULATION DATA --
                # Get data from simulation at current timestep (and calling point)
                self.time = self.bca.get_ems_data(['t_datetimes'])
                self.day_of_week = self.bca.get_ems_data(['t_weekday']) # An integer day of week (1-7)

                self.zn1_temp = self.bca.get_ems_data(['zn1_temp']) # deg C
                self.zn2_temp = self.bca.get_ems_data(['zn2_temp']) # deg C

                    # first element next hour, reverse list for RNN
                self.future_work_hour_booleans = [] # school open? boolean
                # self.future_work_hour_booleans_len = 72

                counter = 0
                while counter < self.future_work_hour_booleans_len:
                    if (self.work_day_start.hour <= (self.time.hour + counter) % 24 < self.work_day_end.hour # check time
                    and ( 0 < (self.day_of_week + math.floor((self.time.hour + counter) / 24)) % 7 < 6) ): # check weekend. Sunday = 0
                        self.future_work_hour_booleans.append(1)
                    else:
                        self.future_work_hour_booleans.append(0)
                    counter += 1

                self.future_global_rad = [] # Global radiation Wh / m2, 0-1000
                # self.future_global_rad_len = 24
                counter = 0
                while counter < self.future_global_rad_len:
                    temp_weather = 0
                    if self.time.hour + counter <= 23: # Today
                        temp_weather = self.bca.get_weather_forecast(['beam_solar'], 'today', self.time.hour + counter, 1) # first timestep is 1

                    elif self.time.hour + counter > 23: # Tomorrow
                        temp_weather = self.bca.get_weather_forecast(['beam_solar'], 'tomorrow', (self.time.hour + counter) % 24, 1) # first timestep is 1

                    self.future_global_rad.append(temp_weather)
                    counter += 1


                self.future_diffuse_rad = [] # diffuse radiation Wh / m2, 0-1000
                # self.future_diffuse_rad_len = 24
                counter = 0
                while counter < self.future_diffuse_rad_len:
                    temp_weather = 0
                    if self.time.hour + counter <= 23: # Today
                        temp_weather = self.bca.get_weather_forecast(['diffuse_solar'], 'today', self.time.hour + counter, 1) # first timestep is 1

                    elif self.time.hour + counter > 23: # Tomorrow
                        temp_weather = self.bca.get_weather_forecast(['diffuse_solar'], 'tomorrow', (self.time.hour + counter) % 24, 1) # first timestep is 1

                    self.future_diffuse_rad.append(temp_weather)
                    counter += 1


                self.future_ext_temp = [] # external temperatures future
                # self.future_ext_temp_len = 24
                counter = 0
                while counter < self.future_ext_temp_len:
                    temp_weather = 0
                    if self.time.hour + counter <= 23: # Today
                        temp_weather = self.bca.get_weather_forecast(['oa_db'], 'today', self.time.hour + counter, 1) # first timestep is 1

                    elif self.time.hour + counter > 23: # Tomorrow
                        temp_weather = self.bca.get_weather_forecast(['oa_db'], 'tomorrow', (self.time.hour + counter) % 24, 1) # first timestep is 1

                    self.future_ext_temp.append(temp_weather)
                    counter += 1





                var_data = self.bca.get_ems_data(self.var_names)
                meter_data = self.bca.get_ems_data(self.meter_names, return_dict=True)
                weather_data = self.bca.get_ems_data(self.weather_names, return_dict=True)  # just for example, other usage

                # Update our MdpManager and all MdpElements, returns same values
                # Automatically runs any encoding functions to update encoded values
                vars = self.mdp.update_ems_value(self.vars, var_data)  # outputs dict based on ordered list of names & values
                meters = self.mdp.update_ems_value_from_dict(meter_data)  # other usage, outputs same dict w/ dif input
                weather = self.mdp.update_ems_value_from_dict(weather_data)   # other usage, outputs same dict w/ dif input

                """
                Below, we show various redundant ways of looking at EMS values and encoded values. A variety of approaches are 
                provided for a variety of use-cases. Please inspect the usage and code to see what best suites your needs. 
                Note: not all usage examples are presented below.
                """
                # Get specific values from MdpManager based on name
                self.zn1_temp = self.mdp.get_mdp_element('zn1_temp').value
                # OR get directly from BcaEnv
                self.zn1_temp = self.bca.get_ems_data(['zn1_temp'])
                # OR directly from output
                self.zn1_temp = var_data[0]  # from BcaEnv dict output
                self.zn1_temp = vars['zn1_temp']  # from MdpManager list output
                # outdoor air dry bulb temp
                outdoor_temp = weather_data['oa_db']  # from BcaEnv dict output
                outdoor_temp = weather['oa_db']  # from MdpManager dict output

                # use encoding function values to see temperature in Fahrenheit
                zn1_temp_f = self.mdp.ems_master_list['zn1_temp'].encoded_value  # access the Master list dictionary directly
                outdoor_temp_f = self.mdp.get_mdp_element('oa_db').encoded_value  # using helper function
                # OR call encoding function on multiple elements, even though encoded values are automatically up to date
                encoded_values_dict = self.mdp.get_ems_encoded_values(['oa_db', 'zn1_temp'])
                zn1_temp_f = encoded_values_dict['zn1_temp']
                outdoor_temp_f = encoded_values_dict['oa_db']

            
                # print reporting
                # if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
                #     print(f'\n\nTime: {str(self.time)}')
                #     print('\n\t* Observation Function:')
                #     print(f'\t\tVars: {var_data}\n\t\tMeters: {meter_data}\n\t\tWeather:{weather_data}')
                #     print(f'\t\tZone0 Temp: {round(self.zn1_temp,2)} C, {round(zn1_temp_f,2)} F')
                #     print(f'\t\tOutdoor Temp: {round(outdoor_temp, 2)} C, {round(outdoor_temp_f,2)} F')
                #     print(f'\t\tNew test, outdoor temp tomorrow at same time: {temp_tmw} C.')
                #     print(f'\t\tNew test, outdoor relative humidity tomorrow at same time: {rh_tmw} %.')
                #     print(f'\t\tNew test, outdoor temp tomorrow at NOON with timestep 1: {temp_tmw_noon} C.')
                #     print(f'\t\tNew test, outdoor temp tomorrow at NOON with timestep 1: {temp_tmw_noon_ts1} C.')
                '''sample output
                        * Observation Function:
                    Vars: [21.757708832346, 34.90428052695303]
                    Meters: {'electricity_facility': 10139533.827672353, 'electricity_heating': 0.0}
                    Weather:{'oa_rh': 44.0, 'oa_db': 18.5, 'oa_pa': 96003.0, 'sun_up': True, 'rain': False, 'snow': False, 'wind_dir': 70.0, 'wind_speed': 7.2}
                    Zone0 Temp: 21.76 C, 21.76 F
                    Outdoor Temp: 18.5 C, 65.3 F'''


            def actuation_function(self):
                
                if name_of_control_this_run == 'EPBaseline': # if EPBaseline
                    results_dict = {}
                    return results_dict

                # print('\tfuture work hour booleans ', self.future_work_hour_booleans)
                # print('\tfuture global radiation ', self.future_global_rad)
                # print('\tfuture diffuse radiation ', self.future_diffuse_rad)
                # print('\tfuture external temperatures ', self.future_ext_temp)
                # print('\telectric  heating ', self.bca.get_ems_data(['electricity_heating']))
                # print('\tZone 1 temp ', self.zn1_temp)
                # print('\tZone 2 temp ', self.zn2_temp)

                # 'EPBaseline':0, #  it's 0, network redundant
                # 'RLBaseNoForesight':3,
                # 'RL24hAllRNN':19,
                # 'RL24hNoSolarRNN':15,
                # 'RL04hAllRNN':19,
                # 'RL04hNoSolarRNN':15,
                # 'RL04hFlatInput':87}



                

                """
                def rnn_reduction(normed_list):
                    torch.manual_seed(150795)
                    normed_rev = list(reversed(normed_list))
                    rnn = nn.RNN(len(normed_rev), 4, 1)
                    normedtensor = torch.tensor(normed_rev)
                    normed2D = torch.unsqueeze(normedtensor, 0)
                    input = torch.FloatTensor(normed2D)
                    h0 = torch.rand(1, 4)
                    output, hn = rnn(input, h0)
                    return output.tolist()[0]
                
                # if 24h foresight
                if name_of_control_this_run == 'RL24hAllRNN' or name_of_control_this_run == 'RL24hNoSolarRNN':
                    future_global_rad_rnn = rnn_reduction(future_global_rad_norm[:24])
                    future_diffuse_rad_rnn = rnn_reduction(future_diffuse_rad_norm[:24])
                    future_ext_temp_rnn = rnn_reduction(future_ext_temp_norm[:24])
                # if 4h foresight
                if name_of_control_this_run == 'RL04hAllRNN' or name_of_control_this_run == 'RL04hNoSolarRNN':
                    future_global_rad_rnn = rnn_reduction(future_global_rad_norm[:4])
                    future_diffuse_rad_rnn = rnn_reduction(future_diffuse_rad_norm[:4])
                    future_ext_temp_rnn = rnn_reduction(future_ext_temp_norm[:4])

                future_works_bool_rnn = rnn_reduction(future_work_hour_booleans_norm[:72])
"""

                #observations normalised, full lists
                future_work_hour_booleans_norm = [float(normalise(x, 0, 1)) for x in self.future_work_hour_booleans]
                future_global_rad_norm = [float(normalise(x, 0, 1000)) for x in self.future_global_rad]
                future_diffuse_rad_norm = [float(normalise(x, 0, 1000)) for x in self.future_diffuse_rad]
                future_ext_temp_norm = [float(normalise(x, -10, 40)) for x in self.future_ext_temp]

                zn1_temp_norm = normalise(self.zn1_temp, -10, 40)
                zn2_temp_norm = normalise(self.zn2_temp, -10, 40)
                elec_heating_norm = normalise(self.bca.get_ems_data(['electricity_heating']), 0, 135_000_000)

                # x 4	Global radiation	[0-1000] Wh/m2
                # x 4	Diffuse Radiation
                # x 4	External temperature [-10-40] C
                # x 4 x RNN 72-4 hourly boolean of school on/off
                # x 2 x Internal temperature now
                # 1 x electricity use now max- 134,320,000


                # 'zn1_Airtemp': [('Zone Mean Air Temperature', zn1)],  # deg C
                # 'zn2_Airtemp': [('Zone Mean Air Temperature', zn2)],  # deg C
                # 'zn1_Radtemp': [('Zone Mean Radiant Temperature', zn1)],  # deg C
                # 'zn2_Radtemp': [('Zone Mean Radiant Temperature', zn2)],  # deg C

                # self.bca.get_ems_data(['t_weekday'])


                additional_input_norm_list = np.concatenate(([float(self.bca.get_ems_data([f'{all_zones_louvre_act_names_list[0]}']))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_heating_sp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_heating_sp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['oa_rh']), 0, 100))],
                                              [float(normalise(self.bca.get_ems_data(['oa_db']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['oa_pa']), 90_000, 110_000))],
                                              [float(self.bca.get_ems_data(['sun_up']))],
                                              [float(self.bca.get_ems_data(['rain']))],
                                              [float(normalise(self.bca.get_ems_data(['wind_dir']), 0, 360))],
                                              [float(normalise(self.bca.get_ems_data(['wind_speed']), 0, 30))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_RH']), 0, 100))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_RH']), 0, 100))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_Airtemp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_Airtemp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_Radtemp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_Radtemp']), -10, 40))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_IntWallMassRate']), -30, 30))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_StairsMassRate']), -30, 30))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_IntWallMassRate']), -30, 30))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_IntWallMassEnergy']), -50_000_000, 50_000_000))],
                                              [float(normalise(self.bca.get_ems_data(['zn1_StairsMassEnergy']), -50_000_000, 50_000_000))],
                                              [float(normalise(self.bca.get_ems_data(['zn2_IntWallMassEnergy']), -50_000_000, 50_000_000))],
                                              [float(normalise(self.bca.get_ems_data(['t_weekday']), 1, 7))],

                                            ))
                # TODO, normalise values in list above
                # flat_list = [item for sublist in additional_input_norm_list for item in sublist]

                # current state as np array, add new items to list
                # if statements to pick correct size and inputs for current network
                # All inputs, 19
                # input_state: workhours, temps, radglo, raddif, flats - full length b4 RNN
                if name_of_control_this_run == 'RL24hAllRNN' or name_of_control_this_run == 'RL04hAllRNN':
                    state_prev = np.concatenate((list(reversed(future_work_hour_booleans_norm)),
                                            list(reversed(future_ext_temp_norm[:control_foresight])),
                                            list(reversed(future_global_rad_norm[:control_foresight])),
                                            list(reversed(future_diffuse_rad_norm[:control_foresight])),
                                            [zn1_temp_norm],
                                            [zn2_temp_norm],
                                            [elec_heating_norm],
                                            additional_input_norm_list)) # nnSizeOBS
                # no Solar, 11 inputs only
                if name_of_control_this_run == 'RL24hNoSolarRNN' or name_of_control_this_run == 'RL04hNoSolarRNN':
                    state_prev = np.concatenate((list(reversed(future_work_hour_booleans_norm)),
                                            list(reversed(future_ext_temp_norm[:control_foresight])),
                                            [zn1_temp_norm], 
                                            [zn2_temp_norm],
                                            [elec_heating_norm],
                                            additional_input_norm_list)) # nnSizeOBS
                # no future knowledge, RLBaseNoForesight
                if name_of_control_this_run == 'RLBaseNoForesight':
                    state_prev = np.concatenate((list(reversed(future_work_hour_booleans_norm)),
                                            [zn1_temp_norm], 
                                            [zn2_temp_norm],
                                            [elec_heating_norm],
                                            additional_input_norm_list)) # nnSizeOBS
                
                # flatinput list, RL04hFlatInput, 87 no inputs total. Lists need to be reversed
                if name_of_control_this_run == 'RL04hFlatInput':
                    state_prev = np.concatenate(([zn1_temp_norm], # the no. items must match the QNet input size
                                            [zn2_temp_norm],
                                            [elec_heating_norm],
                                            future_work_hour_booleans_norm,
                                            future_global_rad_norm[:4],
                                            future_diffuse_rad_norm[:4],
                                            future_ext_temp_norm[:4],
                                            additional_input_norm_list)) # nnSizeOBS

                # assert Qnet_flat_input_size == len(state_prev), "The Qnet input size does not match the state input list length"
                

                # reward as float or int
                # reward[idx -1]
                prev_reward = self.reward_calculation()

                # add missing info to old memory (reward and state)
                if self.n_game_steps >= 1:
                    temp_recall = list(self.memory[-1])
                    temp_recall[2] = prev_reward
                    temp_recall[3] = state_prev
                    self.memory[-1] = tuple(temp_recall)

                # train short memory
                if self.n_game_steps > 1:
                    state_before, action_chosen, reward_given, state_after, game_overs = self.memory[-1] # unpack list
                    self.train_short_memory(state_before, action_chosen, reward_given, state_after, game_overs)

                # train long memory / experience replay
                if  self.n_game_steps > 3 and self.day_of_week == 7 and self.time.hour == 23: # Now every day. Prev. sunday evening experience replay self.day_of_week == 7 and
                    # print('\n\t Experince replay activated, as Sunday evening at 23:00 detected')
                    # print('\n\t Experince replay activated, as Sunday evening at 23:00 detected')
                    # print('\n\t Experince replay activated, as evening at 23:00 detected')
                    self.train_long_memory()


                action = self.get_action(state_prev) #predict action reward
                # the action returns list 0's and one 1, pick the 1 as action
                
                assert sum(action) == 1, "There appears to be more than one action chosen by Q-net"
                    # 18, 18.5, 19, 19.5, 20 win open
                    # 18, 18.5, 19, 19.5, 20 win closed
                    # 5 win closed

                    # 18  win open
                    # 19.25, 19.5, 19.75, 20 win closed
                    # 5 win closed

                if action.index(1) <= 4: # nnSizeOBS
                    # print('Index is: ', action.index(1))
                    heat_setpoint = 18 + 2 / 4 * (action.index(1))
                    win_frac = 0 # closed
                elif action.index(1) >= 5 and action.index(1) <= 9:
                    # print('Index is: ', action.index(1))
                    heat_setpoint = 18 + 2 / 4 * (action.index(1) - 5) # -5 as indices are +5 higher
                    win_frac = 1 # open
                elif action.index(1) == 10:
                    # print('Index is: ', action.index(1))
                    heat_setpoint = 5 # low temp option
                    win_frac = 0 # closed
                else:
                    print('Error, no action chosen')
                    print('Index is: ', action.index(1))

                results_dict = {}
                results_dict['zn1_heating_sp'] = heat_setpoint
                results_dict['zn2_heating_sp'] = heat_setpoint

                for i in range(len(all_zones_louvre_act_names_list)):
                    results_dict[all_zones_louvre_act_names_list[i]] = win_frac
                

                
                line_in_manual_df = []
                # line_in_manual_df.append(state_prev)
                # line_in_manual_df.append(additional_input_norm_list)
                line_in_manual_df.append(self.n_game_steps)
                line_in_manual_df.append(self.day_of_week)
                line_in_manual_df.append(self.time)
                line_in_manual_df.append(self.zn1_temp)
                line_in_manual_df.append(self.zn2_temp)
                line_in_manual_df.append(self.bca.get_ems_data(['electricity_heating']))
                manual_dataframe.append(line_in_manual_df)




                # remember new stuff
                next_reward = None
                next_state = None
                game_over = False
                self.remember(state_prev, action, next_reward, next_state, game_over)

                self.n_game_steps += 1

                # print('Game ', agent.n_games, 'Score ', score, 'Record: ', record)
                # create plot for each day time step? - if self.time.hour % 12 == 0 # twice a day
                        # print reporting
                # if self.time.hour % self.print_every_x_hours == 0 and self.time.minute == 0:
                #     print(f'\n\t time is: ', str(self.time), 'weekday is: ', str(self.day_of_week),
                #         f'\n\t* Actuation Function:'
                #         #   f'\n\t\t*{thermostat_settings}*'
                #         f'\n\t\tHeating Setpoint: {heat_setpoint}'
                #         f'\n\t\tWindow open fraction: {win_frac}'
                #         #   f'\n\t\tCooling Setpoint: {cooling_setpoint}\n'
                #         )
                

                # return final_move # this will be return actuations
                # return dict of next_action
                # return actuation dictionary, referring to actuator EMS variables set
                return results_dict
                


        # Create agent instance
        my_agent = Agent(sim, my_mdp)

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


        my_agent.model.save(f'{name_of_control_this_run}-{unique_building_name}')


        # save agent memory
        # current_directory = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work"
        # current_directory is the path for the buliding and the run name, specified at top
        path_record_name = os.path.join(current_directory, 'agent memory.csv')
        df = pd.DataFrame(my_agent.memory, columns= ["state_prev", "action", "next_reward", "next_state", "game_over"])
        df.to_csv(path_or_buf = path_record_name)


        path_record_loss_list = os.path.join(current_directory, 'agent loss.csv')
        loss_list = my_agent.trainer.loss_record
        dfloss = pd.DataFrame(loss_list)
        dfloss.to_csv(path_or_buf = path_record_loss_list)


        # Manual dataframe
        path_record_manual_dataframe = os.path.join(current_directory, 'manual dataframe.csv')
        manual_df = pd.DataFrame(manual_dataframe)
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


