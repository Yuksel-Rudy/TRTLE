import OrcFxAPI
from OrcFxAPI import Model
import yaml
import numpy as np


class LoadYamlData:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_yaml_data()

    def _load_yaml_data(self):
        with open(self.file_path, 'r') as file:
            return yaml.safe_load(file)


class TrtleFlex(Model):
    def __init__(self, vessel, orcfx_file_path, line_file_path):
        super().__init__()
        self.se_location = None
        self.th_location = None

        self.clump = False
        self.umb = False

        self.moorings = None
        self.clumpings = None
        self.umbs = None

        self.moor_settings = None
        self.clump_settings = None
        self.umb_settings = None

        self.env_settings = None

        # load model
        self.LoadData(orcfx_file_path)
        self.vessel_name = vessel
        self.vessel = self[self.vessel_name]

        # watch-circle-related
        self.wc_degs = None
        self.wc_x_rated = None
        self.wc_y_rated = None

        if line_file_path:
            # initialize model
            self.initialize(line_file_path)
        
    def initialize(self, line_file_path):
        line_handler = LoadYamlData(line_file_path)
        name = line_handler.data['name']

        print(f"getting TRTLE data from input file: {name}")

        # environmental setup
        self.env_settings = {'water_depth': line_handler.data['Env']['water_depth']}
        self.define_env()

        # mooring setup
        self.moor_settings = {'moornum': line_handler.data['Moor']['moornum'],
                              'phi': line_handler.data['Moor']['phi'],
                              'fairlead_horizontal_distance': line_handler.data['Moor']['fairlead_distance'][0][
                                  'horizontal'],
                              'fairlead_vertical_distance': line_handler.data['Moor']['fairlead_distance'][1][
                                  'vertical'],
                              'anchor_horizontal_distance': line_handler.data['Moor']['anchor_distance'][0][
                                  'horizontal'],
                              'anchor_vertical_distance': line_handler.data['Moor']['anchor_distance'][1]['vertical'],
                              'beta': line_handler.data['Moor']['beta'],
                              'line_type': line_handler.data['Moor']['line_type'],
                              'target_segment_length': (line_handler.data['Moor']['target_segment_length'],),
                              }
        self.create_moor()

        # clump setup
        if 'clumps' in line_handler.data['Moor']:
            self.clump = True
            self.clump_settings = {'attachment': line_handler.data['Moor']['clumps']['attachment'],
                                   'name': line_handler.data['Moor']['clumps']['name'],
                                   'mass_ratio': line_handler.data['Moor']['clumps']['mass_ratio'],
                                   'num': line_handler.data['Moor']['clumps']['num'],
                                   'delta_ratio': line_handler.data['Moor']['clumps']['delta_ratio'],
                                   }
            self.create_clump()
            
        # Umbilical setup
        if 'UMB' in line_handler.data:
            self.umb = True
            self.umb_settings = {'umb_phi': line_handler.data['UMB']['umb_phi'],
                                 'bend_stiffener_flange_Z': line_handler.data['UMB']['bend_stiffener_flange_Z'],
                                 'umb_horizontal_distance': line_handler.data['UMB']['umb_horizontal_distance'],
                                 'umb_sectional_lengths': line_handler.data['UMB']['umb_sectional_lengths'],
                                 'umb_sectional_type': line_handler.data['UMB']['umb_sectional_type'],
                                 }
            self.create_umb()

    def reinitialize(self):
        self.define_env()

        # delete objects
        for mooring in self.moorings:
            try:
                self.DestroyObject(mooring)

            except Exception as e:
                pass

        self.moorings = []
        self.create_moor()

        # clumps
        if self.clump:
            for clump in self.clumpings:
                self.DestroyObject(clump)

            self.clumpings = []
            self.create_clump()

        if self.umb:
            self.DestroyObject(self.umbs[0])
            self.umbs = []
            self.create_umb()

    def define_env(self):
        # Water depth
        self.environment.WaterDepth = self.env_settings['water_depth']
    def create_moor(self):
        [Lmin, Lmax] = self.compute_Lmin_Lmax()
        angle_between_cables = (360 / self.moor_settings['moornum'])
        self.moorings = []
        for i in range(self.moor_settings['moornum']):
            moor = self.CreateObject(OrcFxAPI.ObjectType.Line, f"Mooring{i}")
            moor.lineType[0] = self.moor_settings['line_type']
            moor.TargetSegmentLength = self.moor_settings['target_segment_length']
            moor.EndAConnection = self.vessel_name
            moor.EndBConnection = 'Anchored'
            moor.EndAX = self.moor_settings['fairlead_horizontal_distance'] * np.cos(
                np.deg2rad((self.moor_settings['phi'] - 180 - angle_between_cables * i)))
            moor.EndBX = self.moor_settings['anchor_horizontal_distance'] * np.cos(
                np.deg2rad((self.moor_settings['phi'] - 180 - angle_between_cables * i)))
            moor.EndAY = self.moor_settings['fairlead_horizontal_distance'] * np.sin(
                np.deg2rad((self.moor_settings['phi'] - 180 - angle_between_cables * i)))
            moor.EndBY = self.moor_settings['anchor_horizontal_distance'] * np.sin(
                np.deg2rad((self.moor_settings['phi'] - 180 - angle_between_cables * i)))
            moor.EndAZ = self.moor_settings['fairlead_vertical_distance']
            moor.EndBZ = self.moor_settings['anchor_vertical_distance']
            moor.LayAzimuth = self.moor_settings['phi'] - i * angle_between_cables
            moor.length[0] = Lmin + self.moor_settings['beta'] * (
                    Lmax - Lmin)  # Assuming the mooring line has one section
            self.moorings.append(moor)

    def create_clump(self):
        self.clumpings = []
        for i in range(len(self.clump_settings['attachment'])):
            # Create Clump Type
            clumpnme = self.clump_settings['name'][i]
            clump_mass = np.abs(self.clump_settings['mass_ratio'][i]) * (
                    self.moorings[self.clump_settings['attachment'][i]].length[0] * self[
                self.moor_settings['line_type']].MassPerUnitLength)
            clump_volume = (self.clump_settings['mass_ratio'][i] < 0.) * 2 * clump_mass / 1.025 + 0.
            if self.clump_settings['num'][i] == 1:
                clump_delta = 0.0
            else:
                clump_delta = self.moorings[self.clump_settings['attachment'][i]].length[0] / 2 \
                                 / (0.5 * (self.clump_settings['num'][i] - 1)) * \
                                 self.clump_settings['delta_ratio'][i]

            if not clumpnme in self:
                clump = self.CreateObject(OrcFxAPI.ObjectType.ClumpType, clumpnme)
                clump.Mass = clump_mass / self.clump_settings['num'][i]
                clump.Volume = clump_volume / self.clump_settings['num'][i]
                clump.AlignWith = 'Line axes'
                clump.PenWidth = 20  # for better visualization
                self.clumpings.append(clump)

            # Create Clump Attachment
            moor = self.moorings[self.clump_settings['attachment'][i]]
            attachmentz = []

            for j in np.arange(self.clump_settings['num'][i]):
                moor.AttachmentType = (clumpnme,)
                attachmentz.append(moor.length[0] / 2 - 0.5 * float(self.clump_settings['num'][i] - 1) * \
                                   clump_delta + clump_delta * float(j))

            # Convert list to tuple and assign to moor.Attachmentz
            moor.Attachmentz = tuple(attachmentz)

    def create_umb(self):
        umb = self.CreateObject(OrcFxAPI.ObjectType.Line, "umbilical")
        # Global Configuration
        umb.EndAConnection = self.vessel_name
        umb.EndBConnection = 'Anchored'
        # umb.IncludeTorsion = 'Yes'

        # Global configuration settings:
        umb.EndAX = 0.0
        umb.EndAY = 0.0
        umb.EndAZ = 0.0
        umb.EndBX = self.umb_settings['umb_horizontal_distance'] * np.cos(
            np.deg2rad((self.umb_settings['umb_phi'] - 180)))
        umb.EndBY = self.umb_settings['umb_horizontal_distance'] * np.sin(
            np.deg2rad((self.umb_settings['umb_phi'] - 180)))
        umb.EndBZ = 0.0
        umb.LayAzimuth = self.umb_settings['umb_phi']

        # # Set stiffness and twisting at both ends to infinity
        # umb.EndAxBendingStiffness = 5e7
        # umb.EndBxBendingStiffness = 5e7
        # umb.EndAyBendingStiffness = 5e7
        # umb.EndByBendingStiffness = 5e7
        # umb.EndATwistingStiffness = 5e7
        # umb.EndBTwistingStiffness = 5e7

        # Create the different Sections
        linetype_top = ('bend stiffener - steel section', 'bend stiffener - polymer section')
        # linetype_bottom = tuple(reversed(linetype_top))
        line_type_umb = tuple([umb_sec_type for umb_sec_type in self.umb_settings['umb_sectional_type']])
        linetype = linetype_top + line_type_umb
        # Bend Stiffener settings:
        # top
        umb.lineType = linetype
        umb.length[0] = 1 / 6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        umb.length[1] = 5 / 6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        umb.TargetSegmentLength[0] = 1 / 6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        umb.TargetSegmentLength[1] = 1 / 6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        # bottom
        # umb.length[-1] = 1/6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        # umb.length[-2] = 5/6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        # umb.TargetSegmentLength[-1] = 1/6 * abs(self.umb_settings['bend_stiffener_flange_Z'])
        # umb.TargetSegmentLength[-2] = 1/6 * abs(self.umb_settings['bend_stiffener_flange_Z'])

        # Umbilical power cable settings
        i = 2
        for umb_sec_length in self.umb_settings['umb_sectional_lengths']:
            umb.length[i] = umb_sec_length
            umb.TargetSegmentLength[i] = (0.5 / 1e2 * umb_sec_length) * 4
            i += 1

        self.umbs.append(umb)

    def compute_Lmin_Lmax(self):
        # Fairlead Connection
        XF = self.moor_settings['fairlead_horizontal_distance']
        YF = 0.0
        ZF = self.moor_settings['fairlead_vertical_distance']

        # Anchor Connection
        XA = self.moor_settings['anchor_horizontal_distance']
        YA = 0.0
        ZA = -self.environment.WaterDepth + self.moor_settings['anchor_vertical_distance']

        # Minimum and maximum cable length
        Lmin = np.sqrt((XF - XA) ** 2 + (YF - YA) ** 2 + (ZF - ZA) ** 2)
        Lmax = np.abs(XF - XA) + np.abs(YF - YA) + np.abs(ZF - ZA)

        return Lmin, Lmax

    def steel_sectioning(self,
                         top_percentage,
                         bottom_percentage,
                         steellinetype,
                         polyslinetype,
                         target_segment_length):

        # Assuming all mooring lines have the same length:
        Ltot = self.moorings[0].length[0]
        for mooring in self.moorings:
            mooring.LineType = tuple([steellinetype, polyslinetype, steellinetype])
            mooring.Length = tuple([Ltot * top_percentage,
                                   Ltot * (1 - top_percentage - bottom_percentage),
                                   Ltot * bottom_percentage])
            mooring.TargetSegmentLength = tuple([target_segment_length[0], target_segment_length[1], target_segment_length[2]])

    def calculate_se_location(self):
        if self.vessel.IncludeAppliedLoads == 'Yes':
            self.vessel.IncludeAppliedLoads = "No"

        self.CalculateStatics()
        self.se_location = [self.vessel.TimeHistory("X", OrcFxAPI.PeriodNum.StaticState)[0],
                            self.vessel.TimeHistory("Y", OrcFxAPI.PeriodNum.StaticState)[0]]

    def calculate_th_location(self, global_applied_load_origin, global_applied_force, global_applied_moment):
        self.vessel.IncludeAppliedLoads = "Yes"
        self.vessel.GlobalAppliedLoadOriginX = (global_applied_load_origin[0],)
        self.vessel.GlobalAppliedLoadOriginY = (global_applied_load_origin[1],)
        self.vessel.GlobalAppliedLoadOriginZ = (global_applied_load_origin[2],)
        self.vessel.GlobalAppliedForceX = (global_applied_force[0],)
        self.vessel.GlobalAppliedForceY = (global_applied_force[1],)
        self.vessel.GlobalAppliedForceZ = (global_applied_force[2],)
        self.vessel.GlobalAppliedMomentX = (global_applied_moment[0],)
        self.vessel.GlobalAppliedMomentY = (global_applied_moment[1],)
        self.vessel.GlobalAppliedMomentZ = (global_applied_moment[2],)
        self.CalculateStatics()
        self.th_location = [self.vessel.TimeHistory("X", OrcFxAPI.PeriodNum.StaticState)[0],
                            self.vessel.TimeHistory("Y", OrcFxAPI.PeriodNum.StaticState)[0]]
