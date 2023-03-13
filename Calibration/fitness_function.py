

def fitness_calc(kWh_result):
    target_EUI_kWh = 18.22626613

    dif = abs(kWh_result - target_EUI_kWh)

    return round(pow(dif, 2), 4)
