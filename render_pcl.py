#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:46:34 2019

@author: minjie
"""

from skimage.morphology import skeletonize_3d
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


def pcActor(pc, color_pt=(0,0,1), pointSize=0.5):
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()    
    for point in pc:
        pointId = points.InsertNextPoint(point[1], point[2], point[0])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pointId)
    
    vertices.Modified()
    points.Modified()    
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(pointSize)
    actor.GetProperty().SetColor(color_pt)
    actor.GetProperty().SetOpacity(0.5)
    return actor

    
def pcRenderWithSkeleton(pc, pc_skeleton, color_pt=(0,0,1), color_pt_skeleton=(1,0,0),  pointSize=0.5, pointSize_skeleton=2, color_bg=(0.2,0.2,0.2), size_win=(640,640)):
    
    actor = pcActor(pc, color_pt, pointSize)
    actor_s = pcActor(pc_skeleton,color_pt_skeleton, pointSize_skeleton)

    
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
    renderer.AddActor(actor_s)
    renderer.SetBackground(color_bg)
    
    renderWindow.Render()
    m_pInteractorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(m_pInteractorStyle);
    renderWindow.Render()
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()


def text3dActor(text, color_pt=(0,0,0), pointSize=0.5):
    
   
    textSource = vtk.vtkVectorText()
    tt = "(%d,%d,%d)" % (text[0], text[1], text[2])
    textSource.SetText(tt)
    textSource.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(textSource.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.AddPosition(text[1], text[2], text[0])
    actor.SetScale(0.2)
    actor.GetProperty().SetColor(color_pt)
    actor.GetProperty().SetOpacity(0.5)
    return actor


def pcRenderWithSkeletonWithCoord(pc, pc_skeleton, color_pt=(0,0,1), color_pt_skeleton=(1,0,0),  pointSize=0.5, pointSize_skeleton=2, color_bg=(0.2,0.2,0.2), size_win=(640,640)):
    
    actor = pcActor(pc, color_pt, pointSize)
    actor_s = pcActor(pc_skeleton,color_pt_skeleton, pointSize_skeleton)
    
    
    
    
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
    renderer.AddActor(actor_s)
    
    for pp in pc_skeleton:
        actor_text = text3dActor([pp[0],pp[1],pp[2]])
        renderer.AddActor(actor_text)
    
    
    renderer.SetBackground(color_bg)
    
    renderWindow.Render()
    m_pInteractorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(m_pInteractorStyle);
    renderWindow.Render()
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()



def pcRenderWithSegVoxel(anno_yxz_label, pointSize=0.5,  color_bg=(0.2,0.2,0.2), size_win=(1080,1080)):
        # Setup render window, renderer, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(size_win)
    renderWindow.SetWindowName("Point cloud")
    renderWindow.SetPosition(200,0)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.SetBackground(color_bg)
    
    
    n_seg = anno_yxz_label.max()
    for idx in range(1,n_seg+1):
        pc = np.argwhere(anno_yxz_label==idx)
        actor = pcActor(pc, np.random.rand(3), pointSize)

        renderer.AddActor(actor)
    
   
    renderWindow.Render()
    m_pInteractorStyle = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(m_pInteractorStyle);
    renderWindow.Render()
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()



if __name__ == '__main__':
    
    fpath_h5 = 'resources/ct01m_c1.h5' # input h5 file path
    fpath_ply = 'pc1.ply' # output ply file path
    th_conf = 0.25; # confidence threshold
    id_class = 1; # class id
    
    
    with h5py.File(fpath_h5, 'r') as f:
        conf_map = f['label'][...]
    # perform skeletonization
    conf_map1 = skeletonize_3d(conf_map)
    
    
    #conf_map = conf_map_buf[id_class]
    pc_skeleton = np.argwhere(conf_map1 > th_conf)
    #plyWrite(fpath_ply, pc)
    
    pc = np.argwhere(conf_map > th_conf)
    
    color_pt = (0,0,1)
    color_pt_skeleton = (1,0,0)
    color_bg = (0.8,0.8,0.9)
    size_win = (640, 640)
    pointSize = 0.1
    pointSize_skeleton = 4
    
    
    #pcRender(pc,color_pt, pointSize, color_bg,size_win)
    pcRenderWithSkeleton(pc, pc_skeleton, color_pt, color_pt_skeleton, pointSize, pointSize_skeleton, color_bg)