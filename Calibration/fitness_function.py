

def fitness_calc(kWh_result):

    target_kWh = 1000

    dif = abs(kWh_result - target_kWh)

    return round(pow(dif, 2), 2)
