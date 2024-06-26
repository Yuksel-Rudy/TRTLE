from trtle import trtlepy
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
        self.wind_resource_file = None
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
        self.ori_opt1 = None
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

    def create_layout(self, layout_type, layout_properties):
        # turbine selection
        turbine_type = layout_properties["turbine"]

        # initialize first turbine
        if turbine_type == "IEA15MW":
            self.WTG = IEA15MW()

        # farm boundaries
        self.farm_boundaries(layout_properties["boundary_file_path"])

        # energy resources
        self.wind_resource_file = layout_properties["wind_resource_file"]

        # create layout
        if layout_type == "standard":
            self.standard_layout(layout_properties["farm properties"])
        elif layout_type == "honeymooring":
            self.honeymooring_layout(layout_properties["farm properties"])
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

    def standard_layout(self, farm_properties):

        cap = farm_properties["capacity"]  # [MW]
        Dsx = farm_properties["Dspacingx"]  # [-]
        Dsy = farm_properties["Dspacingy"]  # [-]
        ori = farm_properties["orientation"]  # [deg]
        skw = farm_properties["skew factor"]  # [-]
        msr = farm_properties["mooring line spread radius"]  # [m]
        tbl = farm_properties["turbine-boundary limit"]  # [m]

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
        ori_opt1 = +np.ones((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                layout_x[i, j] = xi
                layout_y[i, j] = yi
                if (i + j) % 2 == 0:
                    ori_opt1[i, j] *= - ori_opt1[i, j]


        # Apply theta and skew factor to the generated points
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()
        ori_opt1 = ori_opt1.flatten()
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
        ori_opt1 = ori_opt1[inside]
        # mooring line spread calculation
        # Calculate the distance of each point to the polygon edge
        polygon = Polygon(zip(self.boundary_x, self.boundary_y))
        turbines_with_distances = [(polygon.exterior.distance(Point(x, y)), x, y, idx)
                                   for idx, (x, y) in enumerate(zip(layout_x, layout_y))]

        # Sort the turbines by their distance to the edge (closest first)
        turbines_sorted_by_edge_proximity = sorted(turbines_with_distances, key=lambda x: x[0])

        turbines_msr = [turbine for turbine in turbines_sorted_by_edge_proximity if turbine[0] >= tbl]
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
            layout_x, layout_y, ori_opt1 = zip(*[(x, y, ori_opt1[idx]) for x, y, idx in selected_turbines_sorted_back])
            layout_x, layout_y = np.array(layout_x), np.array(layout_y)
        else:
            ori_opt1 = [ori_opt1[idx] for _, _, _, idx in turbines_msr]

        self.layout_x, self.layout_y, self.ori_opt1 = layout_x, layout_y, ori_opt1
        self.turbine_ct = len(layout_x)
        self.spacing_x = Dsx
        self.spacing_y = Dsy
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        msrs = np.zeros(len(self.layout_x)) + msr
        tbls = np.zeros(len(self.layout_x)) + tbl
        moris = np.zeros(len(self.layout_x))
        self.populate_turbine_keys(tbls, msrs, moris)

    def honeymooring_layout(self, farm_properties):

        cap = farm_properties["capacity"]  # [MW]
        ori = farm_properties["orientation"]  # [deg]
        msr = farm_properties["mooring line spread radius"]  # [m]
        tbl = farm_properties["turbine-boundary limit"]  # [m]

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

        x_even = np.arange((smallest_x - farm_center_x) * magnify,
                           (farm_center_x + largest_x) * magnify,
                           2 * msr * np.cos(np.deg2rad(30)))

        x_odd = x_even + (msr * np.cos(np.deg2rad(30)))

        y = np.arange((smallest_y - farm_center_y) * magnify,
                      (farm_center_y + largest_y) * magnify,
                      1.5 * msr)

        layout_x = np.zeros((2 * len(x_even), len(y)))
        layout_y = np.zeros((2 * len(x_even), len(y)))
        for j, yi in enumerate(y):
            if np.remainder(j, 2) == 0:
                for i, xi in enumerate(x_even):
                    layout_x[i, j] = xi
                    layout_y[i, j] = yi
            else:
                for i, xi in enumerate(x_odd):
                    layout_x[i, j] = xi
                    layout_y[i, j] = yi

        # Apply theta and to the generated points
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()

        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(ori_r), -np.sin(ori_r)],
                                    [np.sin(ori_r), np.cos(ori_r)]])

        # Stack your coordinates in a (2, N) array where N is the number of points
        points = np.vstack((layout_x, layout_y))
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

        turbines_msr = [turbine for turbine in turbines_sorted_by_edge_proximity if turbine[0] >= tbl]
        layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])
        layout_x, layout_y = np.array(layout_x), np.array(layout_y)

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
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        msrs = np.zeros(len(self.layout_x)) + msr
        tbls = np.zeros(len(self.layout_x)) + tbl
        moris = np.zeros(len(self.layout_x))
        self.populate_turbine_keys(tbls, msrs, moris)

    def cluster_layout(self, option=1):
        if option == 1:
            ori_opt1 = np.array(self.ori_opt1)
        else:
            ori_opt1 = -np.array(self.ori_opt1)

        new_msr = np.sqrt((self.spacing_x/2)**2 + (self.spacing_y/2)**2) * self.WTG.diameter()
        for i, turbine in enumerate(self.turbines.values()):
            self.add_update_turbine_keys(i, "msr", new_msr)

        for i, ori1 in enumerate(ori_opt1):
            # Tetra Cluster (Option 1)
            if ori1 == 1.0:
                self.add_update_turbine_keys(i,
                                             "mori",
                                             np.rad2deg(np.arctan2(self.spacing_y, self.spacing_x)) + self.orient)
            else:
                self.add_update_turbine_keys(i,
                                             "mori",
                                             90+np.rad2deg(np.arctan2(self.spacing_x, self.spacing_y)) + self.orient)

    def populate_turbine_keys(self, tbls, msrs, moris):
        for idx, (x, y, tbl, msr, mori) in enumerate(zip(self.layout_x, self.layout_y, tbls, msrs, moris)):
            self.turbines[idx] = {
                'ID': idx,  # turbine ID
                'WTG': self.WTG,  # wind turbine generator type
                'x': x,  # x-location of the turbine
                'y': y,  # y-location of the turbine
                'water_depth': self.calculate_water_depth(x, y),  # water depth (placeholder for now)
                'msr': msr,  # mooring spread radius
                'tbl': tbl,  # turbine-boundary limit
                'mori': mori,
                'pow': self.WTG.power(ws=20) / 1e6  # Power of the turbine at above rated.
            }

    def update_turbine_loc(self):
        for idx, (x, y) in enumerate(zip(self.layout_x, self.layout_y)):
            self.turbines[idx]["x"] = x
            self.turbines[idx]["y"] = y

    def add_update_turbine_keys(self, turbine_id, attribute_name, value):
        if turbine_id in self.turbines:
            self.turbines[turbine_id][attribute_name] = value
        else:
            print(f"Turbine with ID {turbine_id} does not exist.")

    def calculate_water_depth(self, x, y):
        # TODO: write a code to interpolate water depth based on batheymetry file
        water_depth = 800  # Hard-coded
        return water_depth

    def complex_site(self, wind_resource_file_path=None):
        if wind_resource_file_path:
            wind_resources = WindResources(wind_resource_file_path)
        else:
            wind_resources = WindResources(self.wind_resource_file_path)

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

    def collective_watch_circle(self, trtle, delta_theta=45):
        wind_speed = self.WTG.rated_wind_speed
        trtle.calculate_se_location()
        degs = np.arange(0, 360, delta_theta)
        dx = np.zeros(len(degs))
        dy = np.zeros(len(degs))

        print(f"Computing Watch Circle: delta_theta={delta_theta}")
        for i, deg in enumerate(degs):
            print(f"Relocation for deg={deg}")
            wind_direction = deg
            global_applied_load_origin, global_applied_force = self.compute_applied_load(wind_speed,
                                                                                            wind_direction)
            try:
                trtle.calculate_th_location(global_applied_load_origin, global_applied_force, [0., 0., 0.])
                dx[i] = trtle.th_location[0] - trtle.se_location[0]
                dy[i] = trtle.th_location[1] - trtle.se_location[1]
            except Exception as e:
                print(f"could not compute thrust relocation at deg = {deg}")
                print(e)
                dx[i], dy[i] = 0.0, 0.0

        for i, turbine in enumerate(self.turbines.values()):
            # Create the rotation matrix
            theta = np.deg2rad(turbine["mori"])
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

            # Stack your coordinates in a (2, N) array where N is the number of points
            points = np.vstack((dx, dy))
            translated_points = points - np.array([[0.0], [0.0]])

            # Apply the rotation matrix to all points
            rotated_translated_points = np.dot(rotation_matrix, translated_points)
            rotated_points = rotated_translated_points + np.array([[0.0], [0.0]])

            # Split the rotated points back into x and y arrays
            oriented_x = rotated_points[0, :]
            oriented_y = rotated_points[1, :]
            self.add_update_turbine_keys(i, "wc_x", oriented_x)
            self.add_update_turbine_keys(i, "wc_y", oriented_y)
            self.add_update_turbine_keys(i, "wc_d", degs)

    def compute_applied_load(self, wind_speed, wind_direction):
        rho = 1.225
        swept_area = 1 / 4 * np.pi * self.WTG.diameter() ** 2
        CT = np.interp(wind_speed, self.WTG.powerCtFunction.ws_tab, self.WTG.powerCtFunction.power_ct_tab[1])
        thrust_amplitude = 0.5 * rho * swept_area * CT * wind_speed ** 2 / 1e3  # kN
        wind_direction_rad = np.deg2rad(wind_direction)
        global_applied_load_origin = [0., 0., self.WTG.hub_height()]  # m
        global_applied_force = [thrust_amplitude * np.sin(wind_direction_rad),
                                thrust_amplitude * np.cos(wind_direction_rad), 0.]  # kN

        return global_applied_load_origin, global_applied_force

    def anchor_position(self, N_m):
        Ax = np.zeros([N_m, len(self.layout_x)])
        Ay = np.zeros([N_m, len(self.layout_x)])
        # Anchor Positions
        for i, turbine in enumerate(self.turbines.values()):
            for j in range(N_m):
                Ax[j, i] = turbine["x"] + turbine["msr"] * np.cos(
                    np.deg2rad(turbine["mori"] + 360 / N_m * j))
                Ay[j, i] = turbine["y"] + turbine["msr"] * np.sin(
                    np.deg2rad(turbine["mori"] + 360 / N_m * j))
                self.add_update_turbine_keys(i, f"anchor{j}_x", Ax[j, i])
                self.add_update_turbine_keys(i, f"anchor{j}_y", Ay[j, i])

    def anchor_count(self, N_m):
        anchor_dict = {}  # Dictionary to hold anchor positions and counts

        # Iterate over turbines to collect anchor positions
        for i, turbine in enumerate(self.turbines.values()):
            for j in range(N_m):
                # Retrieve the anchor position from the turbine data
                ax = np.round(turbine[f"anchor{j}_x"], 2)
                ay = np.round(turbine[f"anchor{j}_y"], 2)

                if ax is not None and ay is not None:
                    anchor_pos = (ax, ay)

                    # Increment the count for this anchor position, or add it to the dict if not already present
                    if anchor_pos in anchor_dict:
                        anchor_dict[anchor_pos] += 1
                    else:
                        anchor_dict[anchor_pos] = 1

        # Categorize anchors based on how many turbines they are shared with
        shared_anchor_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        for anchor_pos, count in anchor_dict.items():
            if count in shared_anchor_dict:
                shared_anchor_dict[count].append(anchor_pos)

        unique_anchor_count = len(anchor_dict)  # Number of unique anchor positions
        print(f"Anchor count: {unique_anchor_count}")

        sum_by_category = {}
        for count, anchors in shared_anchor_dict.items():
            sum_by_category[count] = len(anchors)  # Count the number of anchors in each category
            print(f"Sum of anchors shared by {count} turbine(s): {sum_by_category[count]}")

        return shared_anchor_dict

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
