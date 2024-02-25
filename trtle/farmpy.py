import pandas as pd
from data.turbines.iea15mw.iea15mw import IEA15MW
import yaml
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import xarray as xr
from py_wake.site import XRSite
from PyWake.py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Niayifar_PorteAgel_2016, Zong_PorteAgel_2020, Blondel_Cathelain_2020
from py_wake.deficit_models import (NOJDeficit,
                                    NOJLocalDeficit,
                                    TurboNOJDeficit,
                                    BastankhahGaussianDeficit,
                                    NiayifarGaussianDeficit,
                                    ZongGaussianDeficit,
                                    CarbajofuertesGaussianDeficit,
                                    TurboGaussianDeficit,
                                    FugaDeficit,
                                    GCLDeficit,
                                    NoWakeDeficit
                                    )


class Farm:
    def __init__(self):
        self.layout_x = None
        self.layout_y = None
        self.turbine = None
        self.site = None
        self.wf_model = None
        self.sim_res = None

    def load_layout_from_file(self, layout_file_path):
        """

        :param layout_file_path: layout file path (first column must have the name `layout_x` and second `layout_y`
        """
        df = pd.read_csv(layout_file_path)
        self.layout_x = list(df["layout_x"])
        self.layout_y = list(df["layout_y"])

    def create_layout(self, boundary_file_path):
        pass

    def turbine_selection(self, turbine_type):
        if turbine_type=="IEA15MW":
            self.turbine = IEA15MW()

    def complex_site(self, wind_resource_file_path):
        wind_resources = WindResources(wind_resource_file_path)
        wd_array = np.array(wind_resources.df_wind["wd"].unique(), dtype=float)
        ws_array = np.array(wind_resources.df_wind["ws"].unique(), dtype=float)
        wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
        freq_interp = NearestNDInterpolator(wind_resources.df_wind[["wd", "ws"]], wind_resources.df_wind["freq_val"])
        freq = freq_interp(wd_grid, ws_grid)
        freq = freq / np.sum(freq)
        ti = wind_resources.turbulence_intensity
        self.site = XRSite(
            ds=xr.Dataset(
                data_vars={'P': (('wd', 'ws'), freq), 'TI': ti},
                coords={'ws': list(ws_array), 'wd': wd_array}))

    def wake_model(self, watch_circle=False):
        self.wf_model = Niayifar_PorteAgel_2016(self.site, self.turbine)
        if watch_circle:
            pass
        else:
            self.sim_res = self.wf_model(self.layout_x,
                                         self.layout_y,
                                         h=None,
                                         type=0,
                                         wd=None,
                                         ws=None,
                                         )
            aep_with_wake = self.sim_res.aep().sum().data
            aep_without_wake = self.sim_res.aep(with_wake_loss=False).sum().data
            wake_effects = (aep_without_wake - aep_with_wake)/(aep_without_wake) * 1e2

            return aep_without_wake, aep_with_wake, wake_effects


class WindResources:

    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            erdata = yaml.safe_load(file)

        # Extracting the wind speed, wind direction, and probability data
        wind_speeds = erdata['wind_resource']['wind_speed']
        wind_directions = erdata['wind_resource']['wind_direction']
        sector_probability = erdata['wind_resource']['sector_probability']['data']
        probabilities = erdata['wind_resource']['probability']['data']
        self.df_wind = pd.DataFrame(columns=['ws', 'wd', 'freq_val'])

        # Loop through all wind speeds and directions to populate the DataFrame
        for i, speed in enumerate(wind_speeds):
            for j, direction in enumerate(wind_directions):
                idx = len(wind_directions) * i + j
                probability = sector_probability[j] * probabilities[j][i]
                self.df_wind.loc[idx] = [speed, direction, probability]

        self.turbulence_intensity = erdata['wind_resource']['turbulence_intensity']['data']
