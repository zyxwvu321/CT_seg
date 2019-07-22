# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import h5py
import numpy as np
import vtk


def plyWrite(fpath_ply, pc):
    f = open(fpath_ply,'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %d\n' % pc.shape[0])
    f.write('property float x\nproperty float y\nproperty float z\n')
    f.write('end_header\n')
    
    for pt in pc:
        z = pt[0]
        x = pt[1]
        y = pt[2]
        f.write('%d %d %d\n' % (x,y,z))
        
    f.write('\n')
    f.close()
    


def pcRender(pc, color_pt=(0,0,1), pointSize=0.5, color_bg=(0.2,0.2,0.2), size_win=(640,640)):
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
#    vtkDepth = vtk.vtkDoubleArray()
    
    for point in pc:
        pointId = points.InsertNextPoint(point[1], point[2], point[0])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, 0)
#        vtkDepth.InsertNextValue(point[2])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pointId)
    
    vertices.Modified()
    points.Modified()
#    vtkDepth.Modified()
    
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
#    polydata.GetPointData().SetScalars(vtkDepth)
    
    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(pointSize)
    actor.GetProperty().SetColor(color_pt)
    
    # Setup render window, renderer, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(size_win)
    renderWindow.SetWindowName("Point cloud")
    renderWindow.SetPosition(200,0)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    
    
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    renderer.SetBackground(color_bg)
    
    renderWindow.Render()
    m_pInteractorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(m_pInteractorStyle);
    renderWindow.Render()
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()




fpath_h5 = './data/h5_rs/p_34.h5'#'resources/ct01m_c1.h5' # input h5 file path



fpath_ply = 'pc1.ply' # output ply file path
th_conf = 0.25; # confidence threshold
id_class = 1; # class id


with h5py.File(fpath_h5, 'r') as f:
    conf_map = f['label'][...]

#conf_map = conf_map_buf[id_class]
pc = np.argwhere(conf_map > th_conf)
#plyWrite(fpath_ply, pc)

color_pt = (0,0,1)
color_bg = (0.8,0.8,0.9)
size_win = (640, 640)
pointSize = 0.1

pcRender(pc,color_pt, pointSize, color_bg,size_win)