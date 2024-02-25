from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine
cp = np.array([0.000000, 0.049361236, 0.224324252, 0.312216418, 0.36009987, 0.38761204, 0.404010164, 0.413979324, 0.420083692, 0.423787764, 0.425977895, 0.427193272, 0.427183505, 0.426860928, 0.426617959, 0.426458783, 0.426385957, 0.426371389, 0.426268826, 0.426077456, 0.425795302, 0.425420049, 0.424948854, 0.424379028, 0.423707714, 0.422932811, 0.422052556, 0.421065815, 0.419972455, 0.419400676, 0.418981957, 0.385839135, 0.335840083, 0.29191329, 0.253572514, 0.220278082, 0.191477908, 0.166631343, 0.145236797, 0.126834289, 0.111011925, 0.097406118, 0.085699408, 0.075616912, 0.066922115, 0.059412477, 0.052915227, 0.04728299, 0.042390922, 0.038132739, 0.03441828, 0.0, 0.0])
v = np.array([0.000, 3, 3.54953237, 4.067900771, 4.553906848, 5.006427063, 5.424415288, 5.806905228, 6.153012649, 6.461937428, 6.732965398, 6.965470002, 7.158913742, 7.312849418, 7.426921164, 7.500865272, 7.534510799, 7.541241633, 7.58833327, 7.675676842, 7.803070431, 7.970219531, 8.176737731, 8.422147605, 8.70588182, 9.027284445, 9.385612468, 9.780037514, 10.20964776, 10.67345004, 10.86770694, 11.17037214, 11.6992653, 12.25890683, 12.84800295, 13.46519181, 14.10904661, 14.77807889, 15.470742, 16.18543466, 16.92050464, 17.67425264, 18.44493615, 19.23077353, 20.02994808, 20.8406123, 21.66089211, 22.4888912, 23.32269542, 24.1603772, 25, 25.020, 50.0])
ct = [0.000000, 0.817533319, 0.792115292, 0.786401899, 0.788898744, 0.790774576, 0.79208669, 0.79185809, 0.7903853, 0.788253035, 0.785845184, 0.783367164, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.77853469, 0.781531069, 0.758935311, 0.614478855, 0.498687801, 0.416354609, 0.351944846, 0.299832337, 0.256956606, 0.221322169, 0.19150758, 0.166435523, 0.145263684, 0.127319849, 0.11206048, 0.099042189, 0.087901155, 0.078337446, 0.07010295, 0.062991402, 0.056831647, 0.05148062, 0.046818787, 0.0, 0.0]
diameter = 242.24
power = 1/2 * cp * 1.255 * (np.pi * (diameter/2)**2) * v**3 / 1e3  # kW

power_curve = np.column_stack((v, power))
ct_curve = np.column_stack((v, ct))


class IEA15MW(WindTurbine):

    def __init__(self, method='linear'):
        u, p = power_curve.T
        WindTurbine.__init__(
            self,
            'IEA15MW',
            diameter=diameter,
            hub_height=150.0,
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w', ct_curve[:, 1], ws_cutin=3, ws_cutout=25,
                                           ct_idle=0.059, method=method))


DTU15WM_RWT = IEA15MW


def main():
    wt = IEA15MW()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    plt.show()


if __name__ == '__main__':
    main()