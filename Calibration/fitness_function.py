

def fitness_calc(kWh_result):

    target_kWh = 2000
    target_EUI = 18.22626613
    # MJ to kWh, 1 MJ = 0.27777777 kWh
    target_MJ = target_EUI / 0.2777777

    dif = abs(kWh_result - target_kWh)

    return round(pow(dif, 2), 2)
