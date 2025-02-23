# from mayavi import mlab

# mlab.figure(bgcolor=(1, 1, 1))
# mlab.test_contour3d()
# mlab.show()

# import vtk
# sphere = vtk.vtkSphereSource()
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(sphere.GetOutputPort())
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)

# renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
# renderWindow = vtk.vtkRenderWindow()
# renderWindow.AddRenderer(renderer)
# renderWindowInteractor = vtk.vtkRenderWindowInteractor()
# renderWindowInteractor.SetRenderWindow(renderWindow)

# renderWindow.Render()
# # renderWindowInteractor.Start()

import wx
app = wx.App(False)
frame = wx.Frame(None, wx.ID_ANY, "Hello World")
frame.Show(True)
app.MainLoop()
