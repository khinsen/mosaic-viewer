# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#       Copyright (C) 2013 The Mosaic Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file LICENSE.txt, distributed as part of this software.
#-----------------------------------------------------------------------------

# Molecular viewer for Mosaic data
#
# Based on the molecular viewer plugin from nMOLDYN.
#
# Authors (in chronological order):
#
#   Eric Pellegrini (ILL)
#   Bachir Aoun (ILL)
#   Gaël Goret (ILL)
#   Konrad Hinsen (CBM)

import gzip
import operator
import sys

import numpy as np

import vtk
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor 

import wx
import wx.aui as aui

import mosaic.api
import mosaic.xml_io

# Rendering modes
LINES = 0
BALLS = 1
STICKS = 2
BALLS_AND_STICKS = 3
BALLS_AND_LINES = 4

# Projections
ORTHOGRAPHIC = 0
PERSPECTIVE = 1

class MolecularViewerError(Exception):
    pass

class MyEventTimer(wx.Timer):
    
    def __init__(self, iren):
        wx.Timer.__init__(self)
        self._iren = iren

    def Notify(self):
        self._iren._Iren.TimerEvent()

class MyRenderWindowInteractor(wxVTKRenderWindowInteractor):
    
    def CreateTimer(self, obj, evt):
        self._timer = MyEventTimer(self)
        self._timer.Start(obj.GetTimerEventDuration(), True)
   
class MolecularViewerPanel(wx.Panel):
    '''
    This class sets up a molecular viewer using vtk functionnalities.
    '''
            
    # Flag defining is universe is contiguous (contiguous is time consuming)
    _contiguous_universe = True
    
    # Rendering mode
    _rendmod = BALLS_AND_STICKS
    
    # The default atom size (in [0,1]).
    _defaultSize = 0.15
    
    # Some predefined size for standard atoms (in [0,1]).
    _sizes = {}
    _sizes["H"] = 0.10
    _sizes["C"] = 0.15
    _sizes["N"] = 0.15
    _sizes["O"] = 0.15
    _sizes["S"] = 0.20
    
    # The color for a selected atom (R,G,B,Alpha).
    __selectionColor = 0
    
    # The default atom color (R,G,B,Alpha).
    _defaultColor = 10 #(215,190,215,255)
    
    # Some predefined colors for standard atoms (R,G,B,Alpha).
    _colors = {}
    _colors["H"]  = 1#(255,255,255,255) 
    _colors["C"]  = 2#(125,249,255,255)
    _colors["N"]  = 3#(0,0,255,255)
    _colors["F"]  = 4#(0.9,0.5,0.5)
    _colors["Cl"] = 5#(0,255,0,255)
    _colors["Br"] = 6#(226,114,91,255)
    _colors["I"]  = 7#(218,165,32,255)
    _colors["O"]  = 8#(255,0,0,255)
    _colors["S"]  = 9#(255,255,0,255)
    
    # Non-atom colors
    _siteLinkColor = 11  # dashed lines linking multiple sites of an atom

    # definition of the lookuptable
    _lut = vtk.vtkColorTransferFunction()
    
    # Some predefined colors for standard atoms in lookuptable (R,G,B,Alpha).
    _lut.AddRGBPoint( 0,  1, 0.2, 1)
    _lut.AddRGBPoint( 1,  1, 1, 1)
    _lut.AddRGBPoint( 2,  0, 1, 0)
    _lut.AddRGBPoint( 3,  0, 0, 1)
    _lut.AddRGBPoint( 4,  0.9, 0.5, 0.5)
    _lut.AddRGBPoint( 5,  0, 1, 0)
    _lut.AddRGBPoint( 6,  1, 0.7, 0.4)
    _lut.AddRGBPoint( 7,  1, 0.4, 0.7)
    _lut.AddRGBPoint( 8,  1, 0, 0)
    _lut.AddRGBPoint( 9,  1, 1, 0)
    _lut.AddRGBPoint(10,  1, 0.9, 0.9)
    _lut.AddRGBPoint(11,  0.2, 0.2, 0.2)
 
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self._parent = parent
        self.build_panel()
        
    def build_panel(self):
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        self._iren = MyRenderWindowInteractor(self, wx.ID_ANY, size=self.GetSize(), flag=wx.EXPAND)
        self._iren.SetPosition((0,0))
        
        sizer.Add(self.iren,1, wx.EXPAND)
        
        self.SetSizer(sizer)        
        sizer.Fit(self)
        self.Layout()

        # define interaction style
        self._iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera() # change interaction style

        self._iren.Enable(1)
        
        # create renderer  
        self._renderer = vtk.vtkRenderer()
        self._iren.GetRenderWindow().AddRenderer(self._renderer)
    
        # cam stuff
        self.camera=vtk.vtkCamera() # create camera 
        self._renderer.SetActiveCamera(self.camera) # associate camera to renderer
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetPosition(0, 0, 20)

        self.__pickerObserverId = None
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        
        self._iren.AddObserver("TimerEvent", self.on_timer)
        self._iren.RemoveObservers("CharEvent")
        self._iren.AddObserver("CharEvent", self.on_keyboard_input)

        self._first = True    
        self._timerCounter = 0
        self._timerInterval = 5
        self._animationLoop = False
        self._universeLoaded = False
        self._currentFrame = 0
        self._maxLaps = 100

    def load_configurations(self, configurations, selection=None):

        self._trajectory = configurations
        self._nFrames = len(configurations)
  
        configuration = configurations[0]
        universe = configuration.universe
        atoms = list(universe.recursive_atom_iterator())
        a_nsites = [a.number_of_sites for a in atoms]
        sites = np.repeat(atoms,
                          [a.number_of_sites for a in atoms])
        a2s = universe.atom_to_site_index_mapping()

        # Display sites rather than atoms and draw bonds between
        # the first sites of each of the two atoms. Link the sites
        # of the same atom by dashed lines.
        bonds = universe.bond_index_array()
        self.bonds = np.take(a2s, bonds.flat).reshape(bonds.shape)
        site_links = []
        for i, ns in enumerate(a_nsites):
            if ns > 1:
                si = a2s[i]
                for j in range(1, ns):
                    site_links.append((si+j-1, si+j))
        self.site_links = np.array(site_links)

        self._nAtoms = len(sites)

        # A normalized atom index array (indices divided by the number of atoms)
        self.scalars = np.arange(self._nAtoms)/float(self._nAtoms)
        
        # The x,y,z values stored in one array
        self.coords = configuration.positions

        # The array that will store the color and alpha scale for all the atoms.
        self.atomColors = np.zeros((len(sites),), np.int) + self._siteLinkColor
        for ai in range(len(atoms)):
            self.atomColors[a2s[ai]] = self._colors.get(atoms[ai].name,
                                                        self._defaultColor)
            
        # The array that will store the size for all the atoms. 
        self.atomSizes = np.array([self._sizes.get(a.name,
                                                   self._defaultSize)
                                   for a in sites])

        # The lattice vectors for drawing a box
        self.lattice = configuration.lattice_vectors()

        try:
            self.clear_universe()
        except:
            self.actor_list = []
        
        
        # Set the configuration and display it.        
        self.set_configuration(0)
        
        self._universeLoaded = True

    @property
    def animation_loop(self):
        return self._animationLoop

    @property
    def current_frame(self):
        return self._currentFrame

    @property
    def iren(self):
        return self._iren

    @property
    def max_laps(self):
        return self._maxLaps

    @property
    def n_frames(self):
        return self._nFrames

    @property
    def n_atoms(self):
        return self._nAtoms

    @property
    def selection_box(self):
        return self.__selectionBox

    @property
    def timer_interval(self):
        return self._timerInterval

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def trajectory_loaded(self):
        return self._universeLoaded

    @property
    def picked_atoms(self):
        return self.__pickedAtoms

    def change_frame_rate(self, laps):
        if not self._universeLoaded:
            return
        
        self._timerInterval = (self._maxLaps - laps)*10 + 1
        if self._animationLoop:
            self._iren.CreateRepeatingTimer(self._timerInterval)

    def create_timer(self):
        self._iren.Initialize()    
        timerId = self._iren.CreateRepeatingTimer(self._timerInterval)
        self._iren.Start()

        return timerId

    def set_frame(self, frame):
        if not self._universeLoaded:
            return
        
        self.stop_animation()
                
        self._timerCounter = frame
        
        self.set_configuration(frame)
                
    def on_keyboard_input(self, obj=None, event=None):
        if not self._universeLoaded:
            return
        
        key = self._iren.GetKeyCode()

        if key in ['1','2','3','4','5']:
            mode = int(key) - 1
            self.set_rendering_mode(mode)

        elif key in ['o','O','p','P']:
            self.set_projection(ORTHOGRAPHIC if key in ['o', 'O']
                                else PERSPECTIVE)
            
        elif key == " ":
            self.start_stop_animation()

    def on_timer(self, obj=None, event=None):
        if self._iren._timer.IsRunning():
            return
        
        self.set_configuration(self._timerCounter)
        self.update_GUI()
        self._timerCounter += 1
    
    def update_GUI(self):
        self._parent.update_frame_number()
        
    def set_rendering_mode(self, mode):
        if not self._universeLoaded:
            return
         
        if self._rendmod != mode:
            self._rendmod=mode
            self.set_configuration(self._currentFrame)

    def set_projection(self, mode):
        if not self._universeLoaded:
            return
        if mode == ORTHOGRAPHIC:
            self.camera.ParallelProjectionOn()
            # Use SetParallelScale to unzoom (factor > 1).
            # The factor should be proportional to the size of the system.
            #self.camera.SetParallelScale(3.) 
        else:
            self.camera.ParallelProjectionOff()
        self._iren.Render()
        
    def goto_first_frame(self):
        if not self._universeLoaded:
            return
        
        self.stop_animation()

        self._timerCounter = 0
        self.set_configuration(0)
        
    def goto_last_frame(self):
        if not self._universeLoaded:
            return
        
        self.stop_animation()
        last = self._nFrames-1
        self._timerCounter = last
        self.set_configuration(last)

    def show_hide_selection_box(self):
        if self._universeLoaded:
            self.__selectionBox.on_off()
        
    def set_timer_interval(self, timerInterval):
        self._timerInterval = timerInterval
        
    def start_animation(self, event=None):
        if self._universeLoaded:
            self._timerId = self.create_timer()
            self._iren.TimerEventResetsTimerOn()
            self._animationLoop = True


    def stop_animation(self, event=None):
        if self._universeLoaded:
            self._iren.TimerEventResetsTimerOff()
            self._animationLoop = False 

    def start_stop_animation(self, event=None, check=True):
        
        if not self._universeLoaded:
            return

        if self._first:
            self._first = False
            
        if not self._animationLoop:
            self.start_animation()
        else:
            self.stop_animation()
    
    def get_renwin(self):
        return self._iren.GetRenderWindow()
        
    def build_polydata(self):   
        '''
        build a vtkPolyData object for a given frame of the trajectory
        '''
        atom_polydata = vtk.vtkPolyData()
        
        coords, _ = ndarray_to_vtkpoints(self.coords)
        atom_polydata.SetPoints(coords)
        
        scalars = ndarray_to_vtkarray(self.atomColors,
                                      self.atomSizes,
                                      self._nAtoms) 
        atom_polydata.GetPointData().SetScalars(scalars) 
        
        bonds = ndarray_to_vtkcellarray(self.bonds)
        atom_polydata.SetLines(bonds)
        rendmod = self._rendmod
        
        self.actor_list = []
        line_actor=None
        ball_actor=None
        tube_actor=None
        
        if rendmod in [LINES, BALLS_AND_LINES] :
            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInput(atom_polydata)
            line_mapper.SetLookupTable(self._lut)
            line_mapper.ScalarVisibilityOn()
            line_mapper.ColorByArrayComponent("scalars", 1)
            line_actor = vtk.vtkActor()
            line_actor.GetProperty().SetLineWidth(3)
            line_actor.SetMapper(line_mapper)
            self.actor_list += [line_actor]
            
        if rendmod in [BALLS, BALLS_AND_STICKS, BALLS_AND_LINES]:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(0.2)
            glyph = vtk.vtkGlyph3D()
            glyph.SetInput(atom_polydata)
            glyph.SetScaleModeToScaleByScalar()
            glyph.SetColorModeToColorByScalar()
            glyph.SetScaleFactor(1)
            glyph.SetSource(sphere.GetOutput())
            glyph.SetIndexModeToScalar()
            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetLookupTable(self._lut)
            sphere_mapper.SetScalarRange(atom_polydata.GetScalarRange())
            sphere_mapper.SetInputConnection(glyph.GetOutputPort())            
            sphere_mapper.ScalarVisibilityOn()
            sphere_mapper.ColorByArrayComponent("scalars", 1)
            ball_actor = vtk.vtkActor()
            ball_actor.SetMapper(sphere_mapper)
            ball_actor.GetProperty().SetAmbient(0.2)
            ball_actor.GetProperty().SetDiffuse(0.5)
            ball_actor.GetProperty().SetSpecular(0.3)
            self.actor_list+=[ball_actor]
            self.glyph = glyph

        if rendmod in [STICKS, BALLS_AND_STICKS] :
            tubes = vtk.vtkTubeFilter()
            tubes.SetInput(atom_polydata)
            tubes.SetNumberOfSides(6)
            if rendmod == 2:
                tubes.CappingOn()
                tubes.SetRadius(0.015)
            else:
                tubes.SetCapping(0)
                tubes.SetRadius(0.01)
            tube_mapper = vtk.vtkPolyDataMapper()
            tube_mapper.SetLookupTable(self._lut)
            tube_mapper.SetInputConnection(tubes.GetOutputPort())
            tube_mapper.ScalarVisibilityOn()
            tube_mapper.ColorByArrayComponent("scalars", 1)
            tube_actor = vtk.vtkActor()
            tube_actor.SetMapper(tube_mapper)
            tube_actor.GetProperty().SetAmbient(0.2)
            tube_actor.GetProperty().SetDiffuse(0.5)
            tube_actor.GetProperty().SetSpecular(0.3)
            self.actor_list+=[tube_actor]
            self.tubes = tubes

        if len(self.site_links) > 0:
            link_polydata = vtk.vtkPolyData()
            link_polydata.SetPoints(coords)
            link_polydata.SetLines(ndarray_to_vtkcellarray(self.site_links))
            link_mapper = vtk.vtkPolyDataMapper()
            link_mapper.SetInput(link_polydata)
            link_actor = vtk.vtkActor()
            link_actor.GetProperty().SetLineWidth(3)
            link_actor.GetProperty().SetLineStipplePattern(0xf0f0)
            link_actor.GetProperty().SetColor(0.3, 0.3, 0.3)
            link_actor.SetMapper(link_mapper)
            self.actor_list += [link_actor]

        if len(self.lattice) > 0:
            lattice_points = [np.zeros((3,), np.float)]
            for lv in self.lattice:
                lattice_points.append(lv)
            for i in range(3):
                for j in range(i+1, 3):
                    lattice_points.append(self.lattice[i]+self.lattice[j])
            lattice_points.append(sum(self.lattice))
            lattice_points = np.array(lattice_points)
            if True:
                # Shift the box to put the origin at the center.
                # This should be made configurable through a menu.
                lattice_points -= 0.5*lattice_points[-1]
            box_points, _ = ndarray_to_vtkpoints(lattice_points)
            box_lines = np.array([[0, 1], [0, 2], [0, 3],
                                  [1, 4], [1, 5], [2, 4],
                                  [2, 6], [3, 5], [3, 6],
                                  [4, 7], [5, 7], [6, 7]])
            box_polydata = vtk.vtkPolyData()
            box_polydata.SetPoints(box_points)
            box_polydata.SetLines(ndarray_to_vtkcellarray(box_lines))
            box_mapper = vtk.vtkPolyDataMapper()
            box_mapper.SetInput(box_polydata)
            box_actor = vtk.vtkActor()
            box_actor.GetProperty().SetLineWidth(3)
            box_actor.GetProperty().SetLineStipplePattern(0xaaaa)
            box_actor.GetProperty().SetColor(0.1, 0.1, 0.4)
            box_actor.SetMapper(box_mapper)
            self.actor_list += [box_actor]

        self.picking_domain = {LINES: line_actor,
                               BALLS: ball_actor,
                               STICKS: tube_actor,
                               BALLS_AND_STICKS: ball_actor,
                               BALLS_AND_LINES: ball_actor} \
                              [rendmod]

        assembly = vtk.vtkAssembly()
        for actor in self.actor_list:
            assembly.AddPart(actor) 
        return assembly

    def set_bonds(self):
        '''
        Sets the bonds list.
        '''
        bonds = []
        for k, v in self.bonds.items():
            for vv in v:
                b = sorted([k,vv])
                if b not in bonds:  
                    bonds.append(b)        
        return bonds

    def clear_universe(self):
        if not hasattr(self, "molecule"):
            return 
        self.molecule.VisibilityOff()
        self.molecule.ReleaseGraphicsResources(self.get_renwin())
        self._renderer.RemoveActor(self.molecule)
        del self.molecule 
        
    def show_universe(self):
        '''
        Update the renderer
        '''
        # deleting old frame
        self.clear_universe()
        
        # creating new polydata
        self.molecule = self.build_polydata()

        # adding polydata to renderer
        self._renderer.AddActor(self.molecule)
        
        # rendering
        self._iren.Render()
        
    def set_configuration(self, frame):
        '''
        Sets a new configuration.
        
        @param frame: the configuration number
        @type frame: integer
        '''
        
        self._currentFrame = frame % len(self._trajectory)
        # The new configuration.
        self.coords = self._trajectory[self._currentFrame].positions
                        
        # Reset the view.                        
        self.show_universe() 
        
def ndarray_to_vtkpoints(array):
    """Create vtkPoints from double array"""
    points = vtk.vtkPoints()
    vtkids = {}
    for i in range(array.shape[0]):
        point = array[i]
        vtkid = points.InsertNextPoint(point[0], point[1], point[2])
        vtkids[vtkid]=i
    return points, vtkids

def ndarray_to_vtkarray(colors, radius, nbat):
    # define the colors
    color_scalars = vtk.vtkFloatArray()
    for c in colors:
            color_scalars.InsertNextValue(c)
    color_scalars.SetName("colors")
    
    # some radii
    radius_scalars = vtk.vtkFloatArray()
    for r in radius:
        radius_scalars.InsertNextValue(r)
    radius_scalars.SetName("radius")
    
    # the original index
    index_scalars = vtk.vtkIntArray()
    for i in range(nbat):
        index_scalars.InsertNextValue(i)
    radius_scalars.SetName("index")
    
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(3)
    scalars.SetNumberOfTuples(radius_scalars.GetNumberOfTuples())
    scalars.CopyComponent(0, radius_scalars ,0 )
    scalars.CopyComponent(1, color_scalars ,0 )
    scalars.CopyComponent(2, index_scalars ,0 )
    scalars.SetName("scalars")
    return scalars 

def ndarray_to_vtkcellarray(array):
    bonds=vtk.vtkCellArray()
    for data in array:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,data[0])
        line.GetPointIds().SetId(1,data[1])
        bonds.InsertNextCell(line)

    return bonds

def get_filename():

    filters = 'XML files (*.xml)|*.xml|Compressed XML files (*.xml.gz)|*.xml.gz|HDF5 file (*.h5)|*.h5|ActivePapers (*.ap)|*.ap|All files (*.*)|*.*'
    
    dialog = wx.FileDialog (None, message = 'Open Trajectory file...', wildcard=filters, style=wx.OPEN)

    if dialog.ShowModal() == wx.ID_CANCEL:
        return ""
    
    return dialog.GetPath()    

def build_axes():
    axes = vtk.vtkAxesActor() #  create axes actor
    axes.SetTotalLength( 10, 10 , 10 )
    axes.SetNormalizedShaftLength( 1, 1, 1 )
    axes.SetNormalizedTipLength( 0, 0, 0 )
    axes.AxisLabelsOff()
    axes.GetXAxisTipProperty().SetColor( 0, 0, 1 )
    axes.GetXAxisShaftProperty().SetColor( 0, 0, 1  )
    axes.GetYAxisTipProperty().SetColor( 1, 1, 1 )
    axes.GetYAxisShaftProperty().SetColor( 1, 1, 1 )
    axes.GetZAxisTipProperty().SetColor( 1, 0, 0 )
    axes.GetZAxisShaftProperty().SetColor( 1, 0, 0 )
    return axes

def scaled_bitmap(bitmapPath, width, height):
    image = wx.ImageFromBitmap(wx.Bitmap(bitmapPath))
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    return wx.BitmapFromImage(image)

class AnimationPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self._parent = parent
        self.build_panel()
        
    def build_panel(self):   
        controlSizer = wx.BoxSizer(wx.HORIZONTAL)

        firstButton = wx.BitmapButton(self, wx.ID_ANY, scaled_bitmap('icons/first.png', 32,32))
        self.playPause = wx.BitmapButton(self, wx.ID_ANY, scaled_bitmap('icons/play.png', 32,32))
        lastButton = wx.BitmapButton(self, wx.ID_ANY, scaled_bitmap('icons/last.png', 32,32))
        self.frameSlider = wx.Slider(self,id=wx.ID_ANY, value=0, minValue=0, maxValue=1, style=wx.SL_HORIZONTAL)
        self.frameEntry = wx.TextCtrl(self,id=wx.ID_ANY,value="0", size=(60,20), style= wx.SL_HORIZONTAL|wx.TE_PROCESS_ENTER)        
        speedBitmap = wx.StaticBitmap(self,-1, scaled_bitmap('icons/clock.png', 42,42))
        self.speedSlider = wx.Slider(self,id=wx.ID_ANY,value=0,minValue=0,maxValue=1,style=wx.SL_HORIZONTAL)
        self.speedSlider.SetRange(0,self._parent.max_laps)
        speed = self._parent.max_laps - self._parent.timer_interval
        self.speedSlider.SetValue(speed)
        self.speedEntry = wx.TextCtrl(self,id=wx.ID_ANY,value=str(speed), size=(60,20), style= wx.SL_HORIZONTAL|wx.TE_PROCESS_ENTER)

        controlSizer.Add(firstButton, 0, wx.LEFT|wx.ALIGN_CENTER_VERTICAL,5)
        controlSizer.Add(self.playPause, 0,  wx.ALIGN_CENTER_VERTICAL)
        controlSizer.Add(lastButton, 0, wx.ALIGN_CENTER_VERTICAL)
        controlSizer.Add((5, -1), 0, wx.ALIGN_RIGHT)

        controlSizer.Add(self.frameSlider, 3, wx.ALIGN_CENTER_VERTICAL)
        controlSizer.Add(self.frameEntry, 0, wx.ALIGN_CENTER_VERTICAL)
        controlSizer.Add((5, -1), 0, wx.ALIGN_RIGHT)

        controlSizer.Add(speedBitmap, 0 , wx.ALIGN_CENTER_VERTICAL) 
        controlSizer.Add((5, -1), 0, wx.ALIGN_RIGHT)

        controlSizer.Add(self.speedSlider, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL) 
        controlSizer.Add(self.speedEntry, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL,5) 
                
        self.SetSizer(controlSizer)
        controlSizer.Fit(self)        
        self.Layout()
                
        self.Bind(wx.EVT_SCROLL, self.on_frame_sliding, self.frameSlider)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_set_frame, self.frameEntry)
        
        self.Bind(wx.EVT_SLIDER, self.on_change_frame_rate, self.speedSlider)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_set_speed, self.speedEntry)
        
        self.Bind(wx.EVT_BUTTON, self.on_start_stop_animation, self.playPause)
        self.Bind(wx.EVT_BUTTON, self.on_goto_first_frame, firstButton)
        self.Bind(wx.EVT_BUTTON, self.on_goto_last_frame, lastButton)
                
    def on_change_frame_rate(self, event=None):
        laps = self.speedSlider.GetValue()
        self._parent.change_frame_rate(laps)
        
        self.speedEntry.SetValue(str(self.speedSlider.GetValue()))        

    def on_frame_sliding(self, event=None):
        
        frame = self.frameSlider.GetValue()
        
        self._parent.set_frame(frame)
        
        frame = self._parent.current_frame
        
        self.frameEntry.SetValue(str(frame))
        self.frameSlider.SetValue(frame)

    def on_goto_first_frame(self, event=None):
        
        self._parent.goto_first_frame()

        self.frameEntry.SetValue(str(0))
        self.frameSlider.SetValue(0)
        
    def on_goto_last_frame(self, event=None):

        self._parent.goto_last_frame()

        self.frameEntry.SetValue(str(self._parent.n_frames-1))
        self.frameSlider.SetValue(self._parent.n_frames-1)

    def on_set_speed(self, event=None):
        self.speedSlider.SetValue(int(self.speedEntry.GetValue()))        
        self._parent.change_frame_rate()
        
    def on_timer(self, frame):
        self.frameEntry.SetValue(str(frame))
        self.frameSlider.SetValue(int(frame))

    def on_set_frame(self, event=None):
        frame = int(self.frameEntry.GetValue())
        self._parent.set_configuration(frame)
        self.frameSlider.SetValue(frame)
    
    def on_start_stop_animation(self, event=None):
        self._parent.start_stop_animation()
        self.on_update_animation_icon()

    def on_update_animation_icon(self):
        if self._parent.animation_loop:
            self.playPause.SetBitmapLabel(scaled_bitmap('icons/pause.png', 32,32))
        else:
            self.playPause.SetBitmapLabel(scaled_bitmap('icons/play.png', 32,32))

    def on_set_up_frame_slider(self):
        self._parent.on_set_up_frame_slider()
        
class MainFrame(wx.Frame):
    
    def __init__(self, parent, idx, title="Molecular viewer", trajectory=None, plugins=None):
        wx.Frame.__init__(self, parent, idx, title, size = (950,700))
        self._trajectory = trajectory
        self.__build_dialog()
        
    def __build_dialog(self):
        
        self._mgr = aui.AuiManager(self)
        
        self._viewer = MolecularViewerPanel(self)
        self._animation = AnimationPanel(self)
        
        self._mgr.AddPane(self._viewer,aui.AuiPaneInfo().Dock().Floatable(False).Center().CloseButton(False).CaptionVisible(False))
        self._mgr.AddPane(self._animation,aui.AuiPaneInfo().Bottom().CaptionVisible(False).MinSize((-1,60)))
        
        self._mgr.Update()
        
        self.SetSizeHints(500,500,2000,2000)
        self.SetSize((950, 700))
        
        self.__build_menu()
                        
    def __build_menu(self):

        mainMenu = wx.MenuBar()

        fileMenu = wx.Menu()
        openFile = fileMenu.Append(wx.ID_ANY, '&Open')
        quitProgram = fileMenu.Append(wx.ID_EXIT, '&Exit')

        mainMenu.Append(fileMenu, 'File')

        self.SetMenuBar(mainMenu)
        
        renderingMenu = wx.Menu()
        item = renderingMenu.Append(wx.ID_ANY, "Line")
        self.Bind(wx.EVT_MENU, lambda event : self.set_rendering_mode(LINES), item)

        item = renderingMenu.Append(wx.ID_ANY, "Stick")
        self.Bind(wx.EVT_MENU, lambda event : self.set_rendering_mode(STICKS), item)

        item = renderingMenu.Append(wx.ID_ANY, "Ball")
        self.Bind(wx.EVT_MENU, lambda event : self.set_rendering_mode(BALLS), item)

        item = renderingMenu.Append(wx.ID_ANY, "Ball and line")
        self.Bind(wx.EVT_MENU, lambda event : self.set_rendering_mode(BALLS_AND_LINES), item)

        item = renderingMenu.Append(wx.ID_ANY, "Ball and stick")
        self.Bind(wx.EVT_MENU, lambda event : self.set_rendering_mode(BALLS_AND_STICKS), item)

        renderingMenu.AppendSeparator()

        item = renderingMenu.Append(wx.ID_ANY, "Orthographic")
        self.Bind(wx.EVT_MENU, lambda event : self.set_projection(ORTHOGRAPHIC), item)
        item = renderingMenu.Append(wx.ID_ANY, "Perspective")
        self.Bind(wx.EVT_MENU, lambda event : self.set_projection(PERSPECTIVE), item)

        mainMenu.Append(renderingMenu, "Rendering")

        helpMenu = wx.Menu()
        helpMenu.Append(wx.ID_ANY, 'About')

        mainMenu.Append(helpMenu, 'Help')
        
        self.Bind(wx.EVT_MENU, self.on_open_file, openFile)
        self.Bind(wx.EVT_MENU, self.on_quit, quitProgram)
        self.Bind(wx.EVT_CLOSE, self.on_quit)
   
    ### Linkage property and function between animation tool and viewer
    @property
    def animation_loop(self):
        return self._viewer.animation_loop
 
    @property
    def current_frame(self):
        return self._viewer.current_frame

    @property
    def max_laps(self):
        return self._viewer.max_laps

    @property
    def n_frames(self):
        return self._viewer.n_frames

    @property
    def timer_interval(self):
        return self._viewer.timer_interval
 
    def set_configuration(self, frame):
        self._viewer.set_configuration(frame)
    
    def on_set_up_frame_slider(self):
        self._animation.frameSlider.SetRange(0,self._viewer.n_frames-1)
        
    def update_frame_number(self):
        self._animation.on_timer(self._viewer.current_frame)
    
    def goto_first_frame(self):
        self._viewer.goto_first_frame()
    
    def goto_last_frame(self):
        self._viewer.goto_last_frame()
        
    def change_frame_rate(self,laps):
        self._viewer.change_frame_rate(laps)
        
    def start_stop_animation(self):
        self._viewer.start_stop_animation()
        
    def set_projection(self, mode):
        self._viewer.set_projection(mode)
        
    def set_rendering_mode(self, mode):
        self._viewer.set_rendering_mode(mode)
        
    def set_frame(self, frame):
        self._viewer.set_frame(frame)

    def on_quit(self, event=None):
        d = wx.MessageDialog(None,
                             'Do you really want to quit ?',
                             'Question',
                             wx.YES_NO|wx.YES_DEFAULT|wx.ICON_QUESTION)
        
        if d.ShowModal() == wx.ID_YES:
            self._viewer.iren.TimerEventResetsTimerOff()
            self.Destroy()        

    def on_open_file(self, event=None):
        self._viewer.stop_animation()
        filename = get_filename()
        extension = os.path.splitext(filename)[1]
        if extension not in [".xml", ".gz", ".h5"]:
            raise MolecularViewerError("Unknown file type %r" % extension)
        try:
            if extension == ".h5":
                configurations = self.choose_configurations(load_HDF5(filename))
            else:
                configurations = self.choose_configurations(load_XML(filename))
        except:
            raise MolecularViewerError("The file %r could not be loaded." % filename)
        self._viewer.load_configurations(configurations)
        self.on_set_up_frame_slider()

    def choose_configurations(self, configurations):
        # Let the user pick a selection one day...
        # For now, use all configurations for an arbitrarily
        # chosen universe
        universe = configurations.keys()[0]
        return configurations[universe]


def load_XML(filename):
    extension = os.path.splitext(filename)[1]
    if extension == ".gz":
        xml_file = gzip.GzipFile(filename)
    else:
        xml_file = open(filename)
    try:
        configurations = {}
        for xml_id, data_item in mosaic.xml_io.XMLReader(xml_file):
            if isinstance(data_item, mosaic.api.MosaicConfiguration):
                conf_list = configurations.get(data_item.universe, [])
                conf_list.append(data_item)
                configurations[data_item.universe] = conf_list
    finally:
        xml_file.close()
    if len(configurations) == 0:
        raise ValueError("no configuration in %s" % filename)
    return configurations

def load_HDF5(filename):
    import mosaic.hdf5
    universe_path = {}
    universe_referenced = {}
    configurations = {}
    # This reads all Mosaic data in the file and keeps all
    # configurations in memory. A more efficient method
    # should be implemented later.
    for path, data_item in mosaic.hdf5.HDF5Store(filename).recursive_iterator():
        if isinstance(data_item, mosaic.api.MosaicUniverse):
            universe_path[id(data_item)] = path
        if isinstance(data_item, mosaic.api.MosaicConfiguration):
            key = id(data_item.universe)
            universe_referenced[key] = True
            upath = universe_path[key]
            conf_list = configurations.get(upath, [])
            conf_list.append(data_item)
            configurations[upath] = conf_list
    delete = []
    for universe_id, path in universe_path.items():
        if not universe_referenced[universe_id]:
            delete.append(universe_id)
    for universe_id in delete:
        del universe_path[universe_id]
    if len(configurations) == 0:
        raise ValueError("no configuration in %s" % filename)
    return configurations

if __name__ == "__main__":

    import os
    configurations = None
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        extension = os.path.splitext(filename)[1]
        if extension == ".xml" \
           or (extension == ".gz" and filename.endswith(".xml.gz")):
            configurations = load_XML(filename)
        elif extension in [".h5", ".ap"]:
            configurations = load_HDF5(filename)

    app = wx.App(False)
    f = MainFrame(None, wx.ID_ANY, "Mosaic Molecular Viewer" )
    if configurations:
        configurations = f.choose_configurations(configurations)
        f._viewer.load_configurations(configurations)
        f.on_set_up_frame_slider()
    f.Show()
    app.MainLoop()
