from shapely.geometry import Point, Polygon
import pandas as pd
from data.turbines.iea15mw.iea15mw import IEA15MW
import matplotlib.path as mpath
import yaml
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import xarray as xr
from py_wake.site import XRSite
from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016


class Farm:
    def __init__(self):
        self.wd = None
        self.ws = None
        self.global_wc_x = None
        self.global_wc_y = None
        self.global_wc_d = None
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
        self.chesswise = None
        self.rowwise = None
        self.colwise = None
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

    def create_layout(self, layout_type, layout_properties, mooring_orientation, trtle, capacity_constraint=True, boundary=None):
        # turbine selection
        turbine_type = layout_properties["turbine"]

        # initialize first turbine
        if turbine_type == "IEA15MW":
            self.WTG = IEA15MW()

        # farm boundaries
        self.farm_boundaries(layout_properties["boundary_file_path"], boundary)

        # energy resources
        self.wind_resource_file = layout_properties["wind_resource_file"]

        # create layout
        if layout_type == "standard":
            self.standard_layout(layout_properties["farm properties"], mooring_orientation, trtle, capacity_constraint)
        elif layout_type == "honeymooring":
            self.honeymooring_layout(layout_properties["farm properties"], mooring_orientation, trtle, capacity_constraint)
        else:
            raise ValueError("The layout type specified is not supported!")
        pass

    def farm_boundaries(self, boundary_file_path, boundary=None):
        if boundary is None:
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

    def standard_layout(self, farm_properties, mooring_orientation, trtle, capacity_constraint):

        cap = farm_properties["capacity"]  # [MW]
        Dsx = farm_properties["Dspacingx"]  # [-]
        Dsy = farm_properties["Dspacingy"]  # [-]
        ori = farm_properties["orientation"]  # [deg]
        skw = farm_properties["skew factor"]  # [-]
        msr = farm_properties["mooring line spread radius"]  # [m]
        tbl = farm_properties["turbine-boundary limit"]  # [m]

        # TODO: why was this new_msr here?

        # new_msr = np.sqrt((Dsx / 2) ** 2 + (Dsy / 2) ** 2) * self.WTG.diameter()
        new_msr = msr

        pow = self.WTG.power(ws=20) / 1e6  # [MW]
        ori_r = np.deg2rad(ori)  # [rad]
        turbine_ct = int(round(cap / pow, 0))

        # farm center
        farm_center_x = self.centroid[0]
        farm_center_y = self.centroid[1]
        smallest_x = min(self.boundary_x)
        largest_x = max(self.boundary_x)
        smallest_y = min(self.boundary_y)
        largest_y = max(self.boundary_y)
        magnify = 1.5

        spacing_x = Dsx * self.WTG.diameter()
        spacing_y = Dsy * self.WTG.diameter()
        x = np.arange(smallest_x / magnify, largest_x * magnify, spacing_x)
        y = np.arange(smallest_y / magnify, largest_y * magnify, spacing_y)

        layout_x = np.zeros((len(x), len(y)))
        layout_y = np.zeros((len(x), len(y)))
        ori_opt0 = +np.ones((len(x), len(y)))
        chesswise = +np.ones((len(x), len(y)))
        rowwise = +np.ones((len(x), len(y)))
        colwise = +np.ones((len(x), len(y)))

        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                layout_x[i, j] = xi
                layout_y[i, j] = yi
                if (i + j) % 2 == 0:
                    chesswise[i, j] *= - chesswise[i, j]
                if j % 2 == 0:
                    rowwise[i, j] *= - rowwise[i, j]
                if i % 2 == 0:
                    colwise[i, j] *= - colwise[i, j]

        # Apply theta and skew factor to the generated points
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()
        chesswise = chesswise.flatten()
        rowwise = rowwise.flatten()
        colwise = colwise.flatten()

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

        # center to the farm
        oriented_x += farm_center_x - np.mean(oriented_x)
        oriented_y += farm_center_y - np.mean(oriented_y)

        layout_points = np.column_stack((oriented_x, oriented_y))
        inside = self.polygon.contains_points(layout_points)
        layout_x = oriented_x[inside]
        layout_y = oriented_y[inside]
        chesswise = chesswise[inside]
        rowwise = rowwise[inside]
        colwise = colwise[inside]

        # mooring line spread calculation
        # Calculate the distance of each point to the polygon edge
        polygon = Polygon(zip(self.boundary_x, self.boundary_y))
        turbines_with_distances = [(polygon.exterior.distance(Point(x, y)), x, y, idx)
                                   for idx, (x, y) in enumerate(zip(layout_x, layout_y))]

        if capacity_constraint:
            # Sort the turbines by their distance to the edge (closest first)
            turbines_sorted_by_edge_proximity = sorted(turbines_with_distances, key=lambda x: x[0])

            turbines_msr = [turbine for turbine in turbines_sorted_by_edge_proximity if turbine[0] >= tbl]
            layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])

        else:
            turbines_msr = [turbine for turbine in turbines_with_distances if turbine[0] >= tbl]
            layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])

        # Maximum capacity
        self.max_capacity = len(layout_x) * pow
        print(f"maximum capacity that can fit in the site is {self.max_capacity} MW ({len(layout_x)} turbines)")

        # turbine count:
        if capacity_constraint:
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
                layout_x, layout_y, chesswise, colwise, rowwise = zip(
                    *[(x, y, chesswise[idx], colwise[idx], rowwise[idx]) for x, y, idx in selected_turbines_sorted_back])
                layout_x, layout_y = np.array(layout_x), np.array(layout_y)
            else:
                chesswise = [chesswise[idx] for _, _, _, idx in turbines_msr]
                rowwise = [rowwise[idx] for _, _, _, idx in turbines_msr]
                colwise = [colwise[idx] for _, _, _, idx in turbines_msr]
                layout_x, layout_y = np.array(layout_x), np.array(layout_y)
        else:
            chesswise = [chesswise[idx] for _, _, _, idx in turbines_msr]
            rowwise = [rowwise[idx] for _, _, _, idx in turbines_msr]
            colwise = [colwise[idx] for _, _, _, idx in turbines_msr]
            layout_x, layout_y = np.array(layout_x), np.array(layout_y)


        self.layout_x, self.layout_y, self.chesswise, self.rowwise, self.colwise = layout_x, layout_y, chesswise, rowwise, colwise
        self.turbine_ct = len(layout_x)
        self.spacing_x = Dsx
        self.spacing_y = Dsy
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        msrs = np.zeros(len(self.layout_x)) + new_msr
        tbls = np.zeros(len(self.layout_x)) + tbl
        moris = np.zeros(len(self.layout_x))
        self.populate_turbine_keys(tbls, msrs, moris)

        # Check if anchors all exist inside the farm
        # TODO: Careful here, N_m = 3 is hardcoded.
        N_m = 3
        self.mooring_standard_layout(N_m=N_m, mooring_orientation=mooring_orientation)
        self.anchor_position(N_m=N_m)

        anchor_points_x, anchor_points_y = [], []
        for i, turbine in enumerate(self.turbines.values()):
            for j in range(N_m):
                anchor_points_x.append(turbine[f"anchor{j}_x"])
                anchor_points_y.append(turbine[f"anchor{j}_y"])

        anchor_points = np.column_stack((anchor_points_x, anchor_points_y))
        inside = self.polygon.contains_points(anchor_points)
        inside_array = np.array(inside)
        inside_reshaped = inside_array.reshape((self.turbine_ct, N_m))
        turbines_to_keep = inside_reshaped.all(axis=1)
        layout_x, layout_y = np.array(layout_x)[turbines_to_keep].tolist(), np.array(layout_y)[
            turbines_to_keep].tolist()
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.chesswise = np.array(self.chesswise)[turbines_to_keep].tolist()
        self.rowwise = np.array(self.rowwise)[turbines_to_keep].tolist()
        self.colwise = np.array(self.colwise)[turbines_to_keep].tolist()
        self.turbine_ct = len(layout_x)
        self.spacing_x = Dsx
        self.spacing_y = Dsy
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        # turbine count (second check):
        print(f"After anchor check: capacity that can fit in the site is {self.capacity} MW ({len(self.layout_x)} turbines)")

        msrs = np.zeros(len(self.layout_x)) + new_msr
        tbls = np.zeros(len(self.layout_x)) + tbl
        moris = np.zeros(len(self.layout_x))
        self.populate_turbine_keys(tbls, msrs, moris)
        self.mooring_standard_layout(N_m=N_m, mooring_orientation=mooring_orientation)
        self.anchor_position(N_m=N_m)

        anchor_points_x, anchor_points_y = [], []
        for i, turbine in enumerate(self.turbines.values()):
            for j in range(N_m):
                anchor_points_x.append(turbine[f"anchor{j}_x"])
                anchor_points_y.append(turbine[f"anchor{j}_y"])

        if capacity_constraint:
            if len(self.layout_x) < turbine_ct:
                raise ValueError("Based on the given farm properties, it is not possible to fit the requested capacity"
                                 " inside the given site boundaries")

    def honeymooring_layout(self, farm_properties, mooring_orientation, trtle, capacity_constraint):

        cap = farm_properties["capacity"]  # [MW]
        ori = farm_properties["orientation"]  # [deg]
        msr = farm_properties["mooring line spread radius"]  # [m]
        tbl = farm_properties["turbine-boundary limit"]  # [m]

        pow = self.WTG.power(ws=20) / 1e6  # [MW]
        ori_r = np.deg2rad(ori)  # [rad]
        turbine_ct = int(round(cap / pow, 0))

        # farm center
        farm_center_x = self.centroid[0]
        farm_center_y = self.centroid[1]
        smallest_x = min(self.boundary_x)
        largest_x = max(self.boundary_x)
        smallest_y = min(self.boundary_y)
        largest_y = max(self.boundary_y)
        magnify = 1.1

        x_even = np.arange(smallest_x / magnify, largest_x * magnify, 2 * msr * np.cos(np.deg2rad(30)))
        x_odd = x_even + (msr * np.cos(np.deg2rad(30)))
        y = np.arange(smallest_y / magnify, largest_y * magnify, 1.5 * msr)

        layout_x = np.zeros((2 * len(x_even), len(y)))
        layout_y = np.zeros((2 * len(x_even), len(y)))
        chesswise = +np.ones((2 * len(x_even), len(y)))
        rowwise = +np.ones((2 * len(x_even), len(y)))
        colwise = +np.ones((2 * len(x_even), len(y)))
        rowwise2 = +np.ones((2 * len(x_even), len(y)))
        rowwise22 = +np.ones((2 * len(x_even), len(y)))
        colwise2 = +np.ones((2 * len(x_even), len(y)))
        colwise22 = +np.ones((2 * len(x_even), len(y)))
        for j, yi in enumerate(y):
            if np.remainder(j, 2) == 0:
                for i, xi in enumerate(x_even):
                    layout_x[i, j] = xi
                    layout_y[i, j] = yi
            else:
                for i, xi in enumerate(x_odd):
                    layout_x[i, j] = xi
                    layout_y[i, j] = yi

            if j % 2 == 0:
                colwise[:, j] *= - colwise[:, j]

            if j % 4 == 0:
                colwise2[:, j] *= - colwise2[:, j]

            if (j + 1) % 4 == 0:
                colwise22[:, j] *= - colwise22[:, j]

        for i in range(2 * len(x_even)):
            if i % 2 == 0:
                rowwise[i, :] *= -rowwise[i, :]
            if i % 4 == 0:
                rowwise2[i, :] *= -rowwise2[i, :]
            if (i + 1) % 4 == 0:
                rowwise22[i, :] *= -rowwise22[i, :]

        for i in range(2 * len(x_even)):
            for j in range(len(y)):
                if (i + j) % 2 == 0:
                    chesswise[i, j] *= - chesswise[i, j]

        # Apply theta and to the generated points
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()
        chesswise = chesswise.flatten()
        rowwise = rowwise.flatten()
        rowwise2 = rowwise2.flatten()
        rowwise22 = rowwise22.flatten()
        colwise = colwise.flatten()
        colwise2 = colwise2.flatten()
        colwise22 = colwise22.flatten()
        filter_out = layout_x != 0
        layout_x = layout_x[filter_out]
        layout_y = layout_y[filter_out]
        chesswise = chesswise[filter_out]
        rowwise = rowwise[filter_out]
        rowwise2 = rowwise2[filter_out]
        rowwise22 = rowwise22[filter_out]
        colwise = colwise[filter_out]
        colwise2 = colwise2[filter_out]
        colwise22 = colwise22[filter_out]
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

        # center to the farm
        oriented_x += farm_center_x - np.mean(oriented_x)
        oriented_y += farm_center_y - np.mean(oriented_y)

        layout_points = np.column_stack((oriented_x, oriented_y))
        inside = self.polygon.contains_points(layout_points)
        layout_x = oriented_x[inside]
        layout_y = oriented_y[inside]
        chesswise = chesswise[inside]
        rowwise = rowwise[inside]
        colwise = colwise[inside]
        rowwise2 = rowwise2[inside]
        colwise2 = colwise2[inside]
        rowwise22 = rowwise22[inside]
        colwise22 = colwise22[inside]
        # mooring line spread calculation
        # Calculate the distance of each point to the polygon edge
        polygon = Polygon(zip(self.boundary_x, self.boundary_y))
        turbines_with_distances = [(polygon.exterior.distance(Point(x, y)), x, y, idx)
                                   for idx, (x, y) in enumerate(zip(layout_x, layout_y))]

        if capacity_constraint:
            # Sort the turbines by their distance to the edge (closest first)
            turbines_sorted_by_edge_proximity = sorted(turbines_with_distances, key=lambda x: x[0])

            turbines_msr = [turbine for turbine in turbines_sorted_by_edge_proximity if turbine[0] >= tbl]
            layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])
            layout_x, layout_y = np.array(layout_x), np.array(layout_y)
        else:
            turbines_msr = [turbine for turbine in turbines_with_distances if turbine[0] >= tbl]
            layout_x, layout_y = zip(*[(x, y) for _, x, y, _ in turbines_msr])
            layout_x, layout_y = np.array(layout_x), np.array(layout_y)

        # Maximum capacity
        self.max_capacity = len(layout_x) * pow
        print(f"maximum capacity that can fit in the site is {self.max_capacity} MW ({len(layout_x)} turbines)")

        if capacity_constraint:
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
                layout_x, layout_y, chesswise, rowwise, colwise, rowwise2, colwise2, rowwise22, colwise22 = \
                    zip(*[(x, y, chesswise[idx], rowwise[idx], colwise[idx], rowwise2[idx], colwise2[idx], rowwise22[idx],
                           colwise22[idx])
                          for x, y, idx in selected_turbines_sorted_back])
                layout_x, layout_y = np.array(layout_x), np.array(layout_y)
            else:
                rowwise = [rowwise[idx] for _, _, _, idx in turbines_msr]
                colwise = [colwise[idx] for _, _, _, idx in turbines_msr]
                rowwise2 = [rowwise2[idx] for _, _, _, idx in turbines_msr]
                colwise2 = [colwise2[idx] for _, _, _, idx in turbines_msr]
                rowwise22 = [rowwise22[idx] for _, _, _, idx in turbines_msr]
                colwise22 = [colwise22[idx] for _, _, _, idx in turbines_msr]
        else:
            rowwise = [rowwise[idx] for _, _, _, idx in turbines_msr]
            colwise = [colwise[idx] for _, _, _, idx in turbines_msr]
            rowwise2 = [rowwise2[idx] for _, _, _, idx in turbines_msr]
            colwise2 = [colwise2[idx] for _, _, _, idx in turbines_msr]
            rowwise22 = [rowwise22[idx] for _, _, _, idx in turbines_msr]
            colwise22 = [colwise22[idx] for _, _, _, idx in turbines_msr]


        self.layout_x, self.layout_y, self.chesswise, self.rowwise, self.colwise, self.rowwise2, self.colwise2, self.rowwise22, self.colwise22 \
            = layout_x, layout_y, chesswise, rowwise, colwise, rowwise2, colwise2, rowwise22, colwise22
        self.turbine_ct = len(layout_x)
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        msrs = np.zeros(len(self.layout_x)) + msr
        tbls = np.zeros(len(self.layout_x)) + tbl
        moris = np.zeros(len(self.layout_x))
        self.populate_turbine_keys(tbls, msrs, moris)

        # Check if anchors all exist inside the farm
        N_m = trtle.moor_settings['moornum']
        self.mooring_honeymooring_layout(N_m, mooring_orientation=mooring_orientation)

        self.anchor_position(N_m=N_m)

        anchor_points_x, anchor_points_y = [], []
        for i, turbine in enumerate(self.turbines.values()):
            for j in range(N_m):
                anchor_points_x.append(turbine[f"anchor{j}_x"])
                anchor_points_y.append(turbine[f"anchor{j}_y"])

        anchor_points = np.column_stack((anchor_points_x, anchor_points_y))
        inside = self.polygon.contains_points(anchor_points)
        inside_array = np.array(inside)
        inside_reshaped = inside_array.reshape((self.turbine_ct, N_m))
        turbines_to_keep = inside_reshaped.all(axis=1)
        layout_x, layout_y = np.array(layout_x)[turbines_to_keep].tolist(), np.array(layout_y)[
            turbines_to_keep].tolist()
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.chesswise = np.array(self.chesswise)[turbines_to_keep].tolist()
        self.rowwise = np.array(self.rowwise)[turbines_to_keep].tolist()
        self.colwise = np.array(self.colwise)[turbines_to_keep].tolist()
        self.rowwise2 = np.array(self.rowwise2)[turbines_to_keep].tolist()
        self.colwise2 = np.array(self.colwise2)[turbines_to_keep].tolist()
        self.rowwise22 = np.array(self.rowwise22)[turbines_to_keep].tolist()
        self.colwise22 = np.array(self.colwise22)[turbines_to_keep].tolist()
        self.turbine_ct = len(layout_x)
        self.orient = ori
        self.capacity = self.turbine_ct * pow

        # turbine count (second check):
        print(f"After anchor check: capacity that can fit in the site is {self.capacity} MW ({len(layout_x)} turbines)")
        if len(self.layout_x) < turbine_ct:
            raise ValueError("Based on the given farm properties, it is not possible to fit the requested capacity"
                             " inside the given site boundaries")

    def cluster_layout(self, farm_properties, trtle):
        pass

    def mooring_standard_layout(self, N_m, mooring_orientation="DMO_01"):
        if N_m == 2 or N_m == 4:
            if mooring_orientation == "DMO_01":  # Dual Mooring Orientation: white-black-white
                ori_opt1 = np.array(self.chesswise)
            elif mooring_orientation == "DMO_02":  # Dual Mooring Orientation: black-white-black
                ori_opt1 = -np.array(self.chesswise)
            elif mooring_orientation == "IMO":  # Identical Mooring Orientation
                ori_opt1 = np.ones_like(self.chesswise)
            for i, ori1 in enumerate(ori_opt1):
                if ori1 == 1.0:
                    self.add_update_turbine_keys(i,
                                                 "mori",
                                                 np.rad2deg(np.arctan2(self.spacing_y, self.spacing_x)) + self.orient)
                else:
                    self.add_update_turbine_keys(i,
                                                 "mori",
                                                 90 + np.rad2deg(
                                                     np.arctan2(self.spacing_x, self.spacing_y)) + self.orient)
        if N_m == 3:
            if mooring_orientation == "DMO_03":  # rowwise variation
                for i, ori3 in enumerate(self.rowwise):
                    self.add_update_turbine_keys(i, "mori", self.orient if ori3 == 1 else self.orient + 180)  #

    def mooring_honeymooring_layout(self, N_m, mooring_orientation="DMO_01"):
        if N_m == 2:
            oris = np.ones_like(self.chesswise)
            colwise2 = np.ones_like(self.colwise2)
            colwise22 = np.ones_like(self.colwise22)
            if mooring_orientation == "DMO_01":
                oris = np.array(self.chesswise)
            elif mooring_orientation == "DMO_02":
                oris = -np.array(self.chesswise)
            elif mooring_orientation == "DMO_03":
                oris = np.array(self.chesswise)
                colwise2 = np.array(self.colwise2)
                colwise22 = np.array(self.colwise22)
            elif mooring_orientation == "DMO_04":
                oris = -np.array(self.chesswise)
                colwise2 = np.array(self.colwise2)
                colwise22 = np.array(self.colwise22)
            for i, ori in enumerate(oris):
                ori *= colwise2[i] * colwise22[i]
                if ori == 1.0:
                    self.add_update_turbine_keys(i,
                                                 "mori",
                                                 (270 + self.orient + 360 / 3) % 360)
                elif ori == -1.0:
                    self.add_update_turbine_keys(i,
                                                 "mori",
                                                 (270 + self.orient - 360 / 3) % 360)
        elif N_m == 3:
            for i, _ in enumerate(self.turbines):
                self.add_update_turbine_keys(i, "mori", (270 + self.orient) % 360)

    def populate_turbine_keys(self, tbls, msrs, moris):
        self.turbines = {}
        for idx, (x, y, tbl, msr, mori) in enumerate(zip(self.layout_x, self.layout_y, tbls, msrs, moris)):
            self.turbines[idx] = {
                'ID': idx,  # turbine ID
                'WTG': self.WTG,  # wind turbine generator type
                'thrust_max': 0.5 * self.WTG.ct(self.WTG.rated_wind_speed) * 1.255 * \
                              (np.pi * (self.WTG.diameter() / 2) ** 2) * self.WTG.rated_wind_speed ** 2 / 1e3,
                # thrust max
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

    def complex_site(self, wind_resource_file=None):
        if wind_resource_file:
            wind_resources = WindResources(wind_resource_file)
        else:
            wind_resources = WindResources(self.wind_resource_file)

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
        self.wd = wd_array
        self.ws = ws_array

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

    def wake_model(self, watch_circle=False, tol=5):
        self.wf_model = Niayifar_PorteAgel_2016(self.site, self.WTG)
        if watch_circle:
            aep_without_wake, aep_with_wake, wake_effects = self.compute_AEP(tol)
        else:
            self.sim_res = self.wf_model(self.layout_x,
                                         self.layout_y,
                                         h=None,
                                         type=0,
                                         wd=self.wd,
                                         ws=self.ws,
                                         )
            aep_with_wake = self.sim_res.aep().sum().data
            aep_without_wake = self.sim_res.aep(with_wake_loss=False).sum().data
            wake_effects = (aep_without_wake - aep_with_wake) / (aep_without_wake) * 1e2

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

        self.global_wc_x = dx
        self.global_wc_y = dy
        self.global_wc_d = degs
        self.update_watch_circle(trtle)

    def update_watch_circle(self, trtle):
        dx = self.global_wc_x
        dy = self.global_wc_y
        degs = self.global_wc_d
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
            self.add_update_turbine_keys(i, "se_location", trtle.se_location)
            self.add_update_turbine_keys(i, "th_location", trtle.th_location)
            self.add_update_turbine_keys(i, "wc_x", oriented_x)
            self.add_update_turbine_keys(i, "wc_y", oriented_y)
            self.add_update_turbine_keys(i, "wc_d", degs)

    def compute_applied_load(self, wind_speed, wind_direction):
        rho = 1.225
        swept_area = 1 / 4 * np.pi * self.WTG.diameter() ** 2
        CT = self.WTG.ct(wind_speed)
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

    def compute_spacing(self):
        # Compute the spacing between the turbines and the angle between the points
        distances = np.zeros((len(self.layout_x), len(self.layout_y)))
        angles = np.zeros((len(self.layout_x), len(self.layout_y)))

        # Calculate pairwise distances and angles for i != j
        for i in range(len(self.layout_x)):
            for j in range(len(self.layout_y)):
                if i != j:
                    x_diff = self.layout_x[j] - self.layout_x[i]
                    y_diff = self.layout_y[j] - self.layout_y[i]
                    distances[i, j] = np.sqrt(x_diff ** 2 + y_diff ** 2)
                    angles[i, j] = np.arctan2(y_diff, x_diff)
                else:
                    distances[i, j] = np.nan
                    angles[i, j] = np.nan

        normalized_spacing = distances / self.WTG.diameter()
        minimum_spacing = np.nanmin(normalized_spacing)
        return normalized_spacing, minimum_spacing, angles

    def compute_AEP(self, tol):
        AEP = 0.0
        wind_speeds = self.sim_res.ws.values
        wind_directions = self.sim_res.wd.values
        freq = self.sim_res.P.values
        for i, wind_speed in enumerate(wind_speeds):
            for j, wind_direction in enumerate(wind_directions):
                [_, _, temp_sim_res] = self.thrust_relocation_loop(wind_direction, wind_speed, tol)
                turbine_powers = temp_sim_res["Power"].values
                AEP += np.sum(turbine_powers / 1e9) * freq[j][i] * 365 * 24

        # Go back to all sim-res
        self.sim_res = self.wf_model(self.layout_x,
                                     self.layout_y,
                                     h=None,
                                     type=0,
                                     wd=self.wd,
                                     ws=self.ws,
                                     )
        aep_without_wake = self.sim_res.aep(with_wake_loss=False).sum().data
        wake_effects = (aep_without_wake - AEP) / (aep_without_wake) * 1e2

        return aep_without_wake, AEP, wake_effects

    def thrust_relocation_loop(self, wind_direction, wind_speed, tol):
        old_layout_x = self.layout_x.copy()
        old_layout_y = self.layout_y.copy()

        new_layout_x = self.layout_x.copy()
        new_layout_y = self.layout_y.copy()

        diff_x = list(np.zeros(len(self.layout_x)))
        diff_y = list(np.zeros(len(self.layout_y)))
        iter = 0
        while True:
            temp_sim_res = self.wf_model(new_layout_x,
                                         new_layout_y,
                                         h=None,
                                         type=0,
                                         wd=wind_direction,
                                         ws=wind_speed,
                                         )
            V = temp_sim_res["WS_eff"].values
            for k, turbine in enumerate(self.turbines.values()):
                V_ijk = V[k][0][0]
                [dx, dy] = self.relocate(turbine, V_ijk, wind_direction)

                rotated_dx = -dx  # need to rotate by 180 after computing from OrcaFlex
                rotated_dy = -dy  # need to rotate by 180 after computing from OrcaFlex
                new_layout_x[k] = self.layout_x[k] + rotated_dx
                new_layout_y[k] = self.layout_y[k] + rotated_dy
                diff_x[k] = new_layout_x[k] - old_layout_x[k]
                diff_y[k] = new_layout_y[k] - old_layout_y[k]
                old_layout_x[k] = new_layout_x[k].copy()
                old_layout_y[k] = new_layout_y[k].copy()

            iter += 1
            if np.mean([np.mean(diff_x), np.mean(diff_y)]) < tol:
                break

            if iter > 1000:
                raise ValueError("relocation could not converge after 1000 iterations!")

        temp_sim_res = self.wf_model(new_layout_x,
                                     new_layout_y,
                                     h=None,
                                     type=0,
                                     wd=wind_direction,
                                     ws=wind_speed,
                                     )
        return new_layout_x, new_layout_y, temp_sim_res

    def relocate(self, turbine, wind_speed, wind_direction):
        '''
        :param turbine: the turbine type class
        :param wind_speed: single wind speed
        :param wind_direction: single wind direction
        :param phi: correction for the phi (mooring configuration)
        :return: new location in x and y based on the wind speed
        '''
        phi = turbine["mori"]
        wind_direction = (wind_direction + phi) % 360
        thrust = 0.5 * turbine["WTG"].ct(wind_speed) * 1.255 * \
                 (np.pi * (turbine["WTG"].diameter() / 2) ** 2) * wind_speed ** 2 / 1e3  # kN
        CT_reduce = thrust / turbine["thrust_max"]
        new_location_x = turbine["se_location"][0] + (
                np.interp(wind_direction, turbine["wc_d"], turbine["wc_x"]) * CT_reduce)
        new_location_y = turbine["se_location"][1] + (
                np.interp(wind_direction, turbine["wc_d"], turbine["wc_y"]) * CT_reduce)

        return new_location_x, new_location_y


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