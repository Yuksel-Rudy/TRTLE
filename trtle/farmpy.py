from shapely.geometry import Point, Polygon
import pandas as pd
from data.turbines.iea15mw.iea15mw import IEA15MW
import matplotlib.path as mpath
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
        self.turbines = {}
        self.max_capacity = None
        self.turbine_ct = None
        self.polygon = None
        self.polygon_points = None
        self.oboundary_x = None
        self.oboundary_y = None
        self.boundary_x = None
        self.boundary_y = None
        self.layout_x = None
        self.layout_y = None
        self.WTG = None
        self.site = None
        self.wf_model = None
        self.sim_res = None
        self.capacity = None
        self.orient = None
        self.spacing_x = None
        self.spacing_y = None

    def load_layout_from_file(self, layout_file_path):
        """

        :param layout_file_path: layout file path (first column must have the name `layout_x` and second `layout_y`
        """
        df = pd.read_csv(layout_file_path)
        self.layout_x = list(df["layout_x"])
        self.layout_y = list(df["layout_y"])

    def create_layout(self, layout_type, layout_properties_file):
        if layout_type=="standard":
            self.standard_layout(layout_properties_file)
        else:
            raise ValueError("The layout type specified is not supported!")
        pass

    def farm_boundaries(self, boundary_file_path):
        boundary = pd.read_csv(boundary_file_path)
        self.boundary_x = list(boundary['boundary_x'])
        self.boundary_y = list(boundary['boundary_y'])
        self.oboundary_x = list(boundary['boundary_x'])
        self.oboundary_y = list(boundary['boundary_y'])
        self.polygon_points = list(zip(self.boundary_x, self.boundary_y))
        self.polygon = mpath.Path(self.polygon_points)
        self.complex_polygon()

    def complex_polygon(self):
        self.centroid = np.mean(self.polygon_points, axis=0)

        def sort_by_angle(point):
            return np.arctan2(point[1] - self.centroid[1], point[0] - self.centroid[0])

        self.polygon_points = sorted(self.polygon_points, key=sort_by_angle)
        self.boundary_x = [point[0] for point in self.polygon_points]
        self.boundary_y = [point[1] for point in self.polygon_points]
        self.polygon = mpath.Path(self.polygon_points)

    def standard_layout(self, layout_properties_file):
        with open(layout_properties_file, 'r') as file:
            layout_data = yaml.safe_load(file)

        # turbine selection
        turbine_type = layout_data["turbine"]

        # initialize first turbine
        if turbine_type == "IEA15MW":
            self.WTG = IEA15MW()

        # farm boundaries
        self.farm_boundaries(layout_data["boundary_file_path"])

        # farm properties
        farm_properties = layout_data["farm properties"]
        cap = farm_properties["capacity"]  # [MW]
        Dsx = farm_properties["Dspacingx"]  # [-]
        Dsy = farm_properties["Dspacingy"]  # [-]
        ori = farm_properties["orientation"]  # [deg]
        skw = farm_properties["skew factor"]  # [-]
        msr = farm_properties["mooring line spread radius"]  # [m]

        pow = self.WTG.power(ws=20) / 1e6  # [MW]
        ori_r = np.deg2rad(ori)  # [rad]
        turbine_ct = int(round(cap/pow, 0))

        # farm center
        farm_center_x = self.centroid[0]
        farm_center_y = self.centroid[1]
        smallest_x = min(self.boundary_x)
        largest_x = max(self.boundary_x)
        smallest_y = min(self.boundary_y)
        largest_y = max(self.boundary_y)
        magnify = 1.1

        spacing_x = Dsx * self.WTG.diameter()
        spacing_y = Dsy * self.WTG.diameter()
        x = np.arange((smallest_x - farm_center_x) * magnify, (farm_center_x + largest_x) * magnify, spacing_x)
        y = np.arange((smallest_y - farm_center_y) * magnify, (farm_center_y + largest_y) * magnify, spacing_y)

        layout_x = np.zeros((len(x), len(y)))
        layout_y = np.zeros((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                layout_x[i, j] = xi
                layout_y[i, j] = yi

        # Apply theta and skew factor to the generated points
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()
        skewed_x_BL = [x + skw * y for x, y in zip(layout_x, layout_y)]
        skewed_y_BL = layout_y  # y-coordinates remain the same

        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(ori_r), -np.sin(ori_r)],
                                    [np.sin(ori_r), np.cos(ori_r)]])

        # Stack your coordinates in a (2, N) array where N is the number of points
        points = np.vstack((skewed_x_BL, skewed_y_BL))
        translated_points = points - np.array([[farm_center_x], [farm_center_y]])

        # Apply the rotation matrix to all points
        rotated_translated_points = np.dot(rotation_matrix, translated_points)
        rotated_points = rotated_translated_points + np.array([[farm_center_x], [farm_center_y]])

        # Split the rotated points back into x and y arrays
        oriented_x = rotated_points[0, :]
        oriented_y = rotated_points[1, :]

        layout_points = np.column_stack((oriented_x, oriented_y))
        inside = self.polygon.contains_points(layout_points)
        layout_x = oriented_x[inside]
        layout_y = oriented_y[inside]

        # mooring line spread calculation
        # Calculate the distance of each point to the polygon edge
        polygon = Polygon(zip(self.boundary_x, self.boundary_y))
        turbines_with_distances = [(polygon.exterior.distance(Point(x, y)), x, y, idx)
                                   for idx, (x, y) in enumerate(zip(layout_x, layout_y))]

        # Sort the turbines by their distance to the edge (closest first)
        turbines_sorted_by_edge_proximity = sorted(turbines_with_distances, key=lambda x: x[0])

        turbines_msr = [turbine for turbine in turbines_sorted_by_edge_proximity if turbine[0] >= msr]
        layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])

        # Maximum capacity
        self.max_capacity = len(layout_x) * pow
        print(f"maximum capacity that can fit in the site is {self.max_capacity} MW ({len(layout_x)} turbines)")

        # turbine count:
        if len(layout_x) < turbine_ct:
            raise ValueError("Based on the given farm properties, it is not possible to fit the requested capacity"
                             " inside the given site boundaries")
        elif len(layout_x) > turbine_ct:
            selected_turbines = turbines_sorted_by_edge_proximity[-turbine_ct:]

            # Extract the x, y coordinates and original indices of the selected turbines
            selected_turbines_with_index = [(x, y, idx) for _, x, y, idx in selected_turbines]

            # Re-sort the selected turbines to their original order
            selected_turbines_sorted_back = sorted(selected_turbines_with_index, key=lambda x: x[2])

            # Extract the re-sorted x and y coordinates
            layout_x, layout_y = zip(*[(x, y) for x, y, _ in selected_turbines_sorted_back])
            layout_x, layout_y = np.array(layout_x), np.array(layout_y)

        self.layout_x, self.layout_y = layout_x, layout_y
        self.turbine_ct = len(layout_x)
        self.spacing_x = Dsx
        self.spacing_y = Dsy
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        msrs = np.zeros(len(self.layout_x)) + msr
        self.populate_turbine_keys(msrs)

    def populate_turbine_keys(self, msrs):
        for idx, (x, y, msr) in enumerate(zip(self.layout_x, self.layout_y, msrs)):
            self.turbines[idx] = {
                'ID': idx,
                'WTG': self.WTG,  # Assuming the same WTG is used for all turbines; adjust if it varies
                'x': x,
                'y': y,
                'water_depth': self.calculate_water_depth(x, y),  # Placeholder for actual calculation
                'msr': msr,
                'pow': self.WTG.power(ws=20) / 1e6  # Example power calculation; adjust as needed
            }

    def add_update_turbine_keys(self, turbine_id, attribute_name, value):
        if turbine_id in self.turbines:
            self.turbines[turbine_id][attribute_name] = value
        else:
            print(f"Turbine with ID {turbine_id} does not exist.")

    def calculate_water_depth(self, x, y):
        water_depth = 800  # Hard-coded
        return water_depth

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

    def polygon_area(self):  # only works for square for now
        """
        Calculate the area of a polygon given its vertices.
        :param vertices: A list of (x, y) tuples representing the vertices of the polygon
        :return: The area of the polygon
        """
        n = len(self.polygon_points)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += self.polygon_points[i][0] * self.polygon_points[j][1]
            area -= self.polygon_points[j][0] * self.polygon_points[i][1]

        self.area = abs(area / 1e6) / 2.0  # km

    def wake_model(self, watch_circle=False):
        self.wf_model = Niayifar_PorteAgel_2016(self.site, self.WTG)
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
