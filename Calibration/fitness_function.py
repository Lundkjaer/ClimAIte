

def fitness_calc(kWh_result):

    target_kWh = 1000

    dif = abs(kWh_result - target_kWh)

    return pow(dif, 2)
