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

import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper_plot import plot

from emspy import EmsPy, BcaEnv, MdpManager


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # a workaround to an error I encountered when running sim  
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


# -- FILE PATHS --
# * E+ Download Path *
ep_path = "C:/EnergyPlusV22-2-0/"  # path to E+ on system / or use V22-2-0/


# IDF File / Modification Paths
idf_file_name = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\Base_model.idf"  # building energy model (BEM) IDF file

# Weather Path
ep_weather_path = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\test_files\GBR_WAL_Lake.Vyrnwy.034100_TMYx.2007-2021.epw"  # EPW weather file

# Output .csv Path (optional)
cvs_output_path = r'C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work\dataframes_output_test.csv'


# def temp_c_to_f(temp_c: float, arbitrary_arg=None):
#     """Convert temp from C to F. Test function with arbitrary argument, for example."""
#     x = arbitrary_arg

#     return 1.8 * temp_c + 32


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

zn0 = 'Z1_Ground_Floor_e5e76ca5' # hint: search .idf for 'spacelist,' or 'zonelist,' this shows the spaces/zones
zn1 = 'Z2_First_Floor_ba8414e7'

tc_intvars = {}

tc_vars = { # These must be present in .idf file and match possible values in .rdd file.
            # example in .idf: Output:Variable,*,Schedule Value,Timestep;
    # Building - same as in .rdd file. rdd file is the variable dictionary.
    # 'hvac_operation_sched': [('Schedule Value', ' SeasonalSchedule_7de19e4d')],  # is building 'open'/'close'? # hint: search .idf for 'schedule:year,' and it will show the name of the schedule, given there is one primary schedule.
    # 'people_occupant_count': [('People Occupant Count', zn0)],  # number of people per Zn0
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': [('Zone Air Temperature', zn0)],  # deg C
    'zn0_RH': [('Zone Air Relative Humidity', zn0)],  # %RH
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
    'wind_speed': [('wind_speed')]  # m/s
    # 'wind_speed_tomorrow': [('wind_speed'), 'tomorrow', 12, 1]  # m/s
} # 'beam_solar', 'diffuse_solar' # Wh/m2

# ACTION SPACE
"""
NOTE: only zn0 (CoreZn) has been setup in the model to allow 24/7 HVAC setpoint control. Other zones have default
HVAC operational schedules and night cycle managers that prevent EMS Actuator control 24/7. Essentially, at times the 
HVAV is "off" and can't be operated. If all zones are to be controlled 24/7, they must be implemented as CoreZn.
See the "HVAC Systems" tab in OpenStudio to zone configurations.
"""
tc_actuators = { # 'user_var_name': ['component_type', 'control_type', 'actuator_key'] within the dict
    # HVAC Control Setpoints
    # 'zn0_cooling_sp': [('Zone Temperature Control', 'Cooling Setpoint', zn0)],  # deg C
    'zn0_heating_sp': [('Zone Temperature Control', 'Heating Setpoint', zn0)],  # deg C
} # Z2_LOUVRE_20,AirFlow Network Window/Door Opening,Venting Opening Factor,[Fraction]
# opening window from .edd APERTURE_E02C3A1A_OPENING,Zone Ventilation,Air Exchange Flow Rate,[m3/s]

# There's the program setpoints too as; PROGRAMTYPE_E5AC29A0_8EC855C7_SETPOINT_HTGSETP,Schedule:Year,Schedule Value,[ ]
# There's also outdoor air, maybe for window openings? MODEL OUTDOOR AIR NODE,System Node Setpoint,Temperature Minimum Setpoint,[C]


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
LR = 0.001

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
        self.zn0_temp = None  # deg C
        self.zn1_temp = None  # deg C
        self.time = None
        self.day_of_week = None
        self.work_hour_booleans_72 = [] # last element is next hour. Reversed list for RNN


        # self.rl_input_params = [self.zn0_temp, # the no. items must match the QNet input size
        #                         self.time.hour,
        #                         self.day_of_week]

        self.work_hours_heating_setpoint = 20  # deg C
        self.work_hours_cooling_setpoint = 25  # deg C
        self.off_hours_heating_setpoint = 15  # deg C # not currently used
        self.off_hours_cooilng_setpoint = 30  # deg C # not currently used
        self.work_day_start = datetime.time(8, 0)  # day starts 6 am
        self.work_day_end = datetime.time(16, 0)  # day ends at 8 pm

        # print reporting
        self.print_every_x_hours = 2

        # DRL
        self.n_game_steps = 0
        self.epsilon = 0 # randomness, greedy/exploration
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if memory larger, it calls popleft()
        self.model = Linear_QNet(4,250,250,250,1) # neural network (input, hidden x3, output)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    
    def get_state(self, game): #observation in BcaEnv
        pass

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) #deque will popleft() if max_memory is reached. Extra () parantheses to store values as single tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples
        else:
           mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) # unpack into lists rather than combined tuples
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def reward_calculation(self): # TODO add in negative reward for spending kWh
        total_reward = 0
        # TODO make temp scale piecewise and increasing
        # get meter reading for prev kwh spending
        # can use electricity_facility? - this seems to be instantenous
        # total_reward -= prev_kwh * 2
        total_reward -= round(self.bca.get_ems_data(['electricity_heating']) / 6_000_000, 2)

        if self.work_day_start <= self.time.time() < self.work_day_end:  #
            # during workday
            if min(self.zn0_temp, self.zn1_temp) < self.work_hours_heating_setpoint:
                total_reward -= 10 * ( self.work_hours_heating_setpoint - min(self.zn0_temp, self.zn1_temp) )

            if max(self.zn0_temp, self.zn1_temp) > self.work_hours_cooling_setpoint:
                total_reward -= 10 * - ( self.work_hours_cooling_setpoint - max(self.zn0_temp, self.zn1_temp) )
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
        self.epsilon = 6000 - self.n_game_steps
        final_move = [0] # [0,0,0] in snake game TODO
        if random.randint(0,12000) < self.epsilon:
            move = random.randint(0,2) # from snake with argmax
            final_move[move] = 1
            # final_move = random.randint(-20,30)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # picks output with highest predicted reward
            final_move[move] = 1 # from snake with argmax
            # final_move[0] = prediction
        
        return final_move

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        # Get data from simulation at current timestep (and calling point)
        self.time = self.bca.get_ems_data(['t_datetimes'])
        self.day_of_week = self.bca.get_ems_data(['t_days']) # An integer day of week (1-7)

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
        self.zn0_temp = self.mdp.get_mdp_element('zn0_temp').value
        # OR get directly from BcaEnv
        self.zn0_temp = self.bca.get_ems_data(['zn0_temp'])
        # OR directly from output
        self.zn0_temp = var_data[0]  # from BcaEnv dict output
        self.zn0_temp = vars['zn0_temp']  # from MdpManager list output
        # outdoor air dry bulb temp
        outdoor_temp = weather_data['oa_db']  # from BcaEnv dict output
        outdoor_temp = weather['oa_db']  # from MdpManager dict output

        # use encoding function values to see temperature in Fahrenheit
        zn0_temp_f = self.mdp.ems_master_list['zn0_temp'].encoded_value  # access the Master list dictionary directly
        outdoor_temp_f = self.mdp.get_mdp_element('oa_db').encoded_value  # using helper function
        # OR call encoding function on multiple elements, even though encoded values are automatically up to date
        encoded_values_dict = self.mdp.get_ems_encoded_values(['oa_db', 'zn0_temp'])
        zn0_temp_f = encoded_values_dict['zn0_temp']
        outdoor_temp_f = encoded_values_dict['oa_db']

        # test TODO remove
        temp_tmw, rh_tmw = self.bca.get_weather_forecast(['oa_db', 'oa_rh'], 'tomorrow', self.time.hour, 1)
        temp_tmw_noon = self.bca.get_weather_forecast(['oa_db'], 'tomorrow', 12, 1)
        temp_tmw_noon_ts1 = self.bca.get_weather_forecast(['oa_db'], 'tomorrow', 12, 1)

        # print reporting
        if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
            print(f'\n\nTime: {str(self.time)}')
            print('\n\t* Observation Function:')
            print(f'\t\tVars: {var_data}\n\t\tMeters: {meter_data}\n\t\tWeather:{weather_data}')
            print(f'\t\tZone0 Temp: {round(self.zn0_temp,2)} C, {round(zn0_temp_f,2)} F')
            print(f'\t\tOutdoor Temp: {round(outdoor_temp, 2)} C, {round(outdoor_temp_f,2)} F')
            print(f'\t\tNew test, outdoor temp tomorrow at same time: {temp_tmw} C.')
            print(f'\t\tNew test, outdoor relative humidity tomorrow at same time: {rh_tmw} %.')
            print(f'\t\tNew test, outdoor temp tomorrow at NOON with timestep 1: {temp_tmw_noon} C.')
            print(f'\t\tNew test, outdoor temp tomorrow at NOON with timestep 1: {temp_tmw_noon_ts1} C.')
            

            '''sample output
                    * Observation Function:
                Vars: [21.757708832346, 34.90428052695303]
                Meters: {'electricity_facility': 10139533.827672353, 'electricity_heating': 0.0}
                Weather:{'oa_rh': 44.0, 'oa_db': 18.5, 'oa_pa': 96003.0, 'sun_up': True, 'rain': False, 'snow': False, 'wind_dir': 70.0, 'wind_speed': 7.2}
                Zone0 Temp: 21.76 C, 21.76 F
                Outdoor Temp: 18.5 C, 65.3 F'''

    def actuation_function(self):
        
        #observations normalised? TODO
        
        temp_tmw = self.bca.get_weather_forecast(['oa_db'], 'tomorrow', self.time.hour, 1) # note: first timestep is 1

        # current state as np array, add new items to list
        state_prev = np.array([self.zn0_temp, # the no. items must match the QNet input size
                                self.time.hour,
                                self.day_of_week,
                                temp_tmw], dtype=float) 
        
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
        if self.day_of_week == 7 and self.time.hour == 23: # sunday evening experience replay
            print('\n\t Experince replay activated, as Sunday evening at 23:00 detected')
            self.train_long_memory()

        action = self.get_action(state_prev) #predict action # need to fix def to handle multiple

        # denormalize actions? no
        # heating_setpoint, second_actuation = zip(action)
        heating_setpoint = round( float(action[0]) , 1) # TODO, might be wrong. It is, this was based on reward = temperature
        
        # remember new stuff
        next_reward = None
        next_state = None
        game_over = False
        self.remember(state_prev, action, next_reward, next_state, game_over)

        self.n_game_steps += 1

        # print('Game ', agent.n_games, 'Score ', score, 'Record: ', record)
        # create plot for each day time step? - if self.time.hour % 12 == 0 # twice a day
                # print reporting
        if self.time.hour % self.print_every_x_hours == 0 and self.time.minute == 0:
            print(f'\n\t* Actuation Function:'
                #   f'\n\t\t*{thermostat_settings}*'
                  f'\n\t\tHeating Setpoint: {heating_setpoint}'
                #   f'\n\t\tCooling Setpoint: {cooling_setpoint}\n'
                  )
        


        # return final_move # this will be return actuations
        # return dict of next_action
        # return actuation dictionary, referring to actuator EMS variables set
        return {
            'zn0_heating_sp': heating_setpoint,
            # 'zn0_cooling_sp': cooling_setpoint
        }
        


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


my_agent.model.save()

# save agent memory
current_directory = r"C:\Users\sebas\Documents\GitHub\ClimAIte\EMS_work"
path_record_name = os.path.join(current_directory, 'agent memory.csv')
df = pd.DataFrame(my_agent.memory, columns= ["state_prev", "action", "next_reward", "next_state", "game_over"])
df.to_csv(path_or_buf = path_record_name)


# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too

# -- Plot Results --
# fig, ax = plt.subplots()
# output_dfs['var'].plot(y='zn0_temp', use_index=True, ax=ax)
# output_dfs['weather'].plot(y='oa_db', use_index=True, ax=ax)
# output_dfs['meter'].plot(y='electricity_HVAC', use_index=True, ax=ax, secondary_y=True)
# output_dfs['actuator'].plot(y='zn0_heating_sp', use_index=True, ax=ax)
# output_dfs['actuator'].plot(y='zn0_cooling_sp', use_index=True, ax=ax)
# plt.title('Zn0 Temps and Thermostat Setpoint for Year')

# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes


