import gmsh
import datetime
import numpy as np
import trimesh
import pyransac3d as pyrsc
from classifier import CLassifier
import pandas as pd
import cylFitting


def getABQFromStpFile(file, elmin=0.6, elmax=6,order=2, material='al'):

    dateStamp = datetime.datetime.now().date().isoformat()
    fn = file.split('.')[:-1][0] + dateStamp
    fn = fn.replace(' ', '_')

    gmsh.initialize()
    gmsh.open(file)

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.option.setNumber("Mesh.MeshSizeMin", elmin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", elmax)

    gmsh.model.mesh.generate(order)



    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodesByElementType(2)

    vertices=np.array([[coord[i],coord[i+1],coord[i+2]] for i in range(0,len(coord),3)])
    faces=[[i,i+1,i+2] for i in range(0,len(vertices),3)]


    np.save('vertices_HAQL.npy', vertices)
    np.save('nodeTags_HAQL.npy', nodeTags)


    gmsh.finalize()


    '--> Check model alignment and orientation'
    '--> translate model in local position'

    mesh = trimesh.Trimesh(vertices=vertices,
                                faces=faces)
    mat={
        'al': [2.7e-3],
        'st': [7.85e-3]
    }

    weight= mesh.volume * mat[material][0]
    #print(mesh.as_open3d)

    caType, recom, features = getRecommendation(mesh.as_open3d)

    if caType == '2 Pt. Guiding Control Arm':
        print(f'Es ist ein {caType}\n'
              f'Fitting possible')
        nodeTagsInner, nodeTagsOuter = cylFitting.getCylinderNodes(vertices, nodeTags)

    else:
        print(f'Es ist ein {caType}\n'
              f'Fitting is not possible')
        nodeTagsInner, nodeTagsOuter= [None], [None]




    #print (nodeTags[:10],vertices[:10])
    
    # msh.show()
    # gmsh.write('{}.inp'.format(fn))
    # gmsh.fltk.run()

    return recom, caType, weight, nodeTagsInner, nodeTagsOuter, features

def twoPointCACylinderFitting(vertices, nodeTags):
    faces=[[i,i+1,i+2] for i in range(0,len(vertices),3)]

    n = int(len(vertices) * 0.25)
    p=vertices[:, 1].argsort()

    vert_o=vertices[p][:n]
    vert_i=vertices[p][-n:]

    sph = pyrsc.Sphere()

    c_o,r,inliners = sph.fit(vert_o, thresh=1.0, maxIteration=1000)
    c_i,r,inliners = sph.fit(vert_i, thresh=1.0, maxIteration=1000)

    mesh=trimesh.Trimesh(vertices=vertices, faces=faces)

    r_o = mesh.nearest.on_surface([c_o])[1][0]
    r_i = mesh.nearest.on_surface([c_i])[1][0]

    inliners_o=fetchCylPts(vert_o, r_o, c_o)
    inliners_i=fetchCylPts(vert_i, r_i, c_i)

    return nodeTags[p][:n][inliners_o], nodeTags[p][-n:][inliners_i], [c_o,c_i],vert_o[inliners_o], vert_i[inliners_i]


def runtimeCheck(minmax = [[0.5, 5], [0.6, 6], [0.7, 7], [0.8, 8], [0.9, 9], [1, 10]]):
    from timeit import default_timer as timer

    times = []

    for element in minmax:
        start = timer()
        getABQFromStpFile('C:/Users/z120768/Desktop/Jupyter_Notebooks/6883115_ZB_LI_QUERLENKER_OBEN_G05.stp',
                          element[0], element[1])
        end = timer()
        times.append([end - start])

    return times

def getRecommendation(mesh_o3d):

    objectdf, features = CLassifier().getObjectFeatureDf(mesh_o3d)

    recom = CLassifier().getResultDf(objectdf, 50, "metadataMoments.csv")

    type= recom['Type'].value_counts().idxmax()
    recom = recom[recom['Type'] == type]

    return type, recom[["FileName","PDM Document No","Weight", "Type", "Distance"]], features



def fetchCylPts(pts, r, c):
    dist = [np.linalg.norm(pts[i] - c) for i in range(0, len(pts))]
    idx = [i for i in range(0, len(pts))]
    p = np.array(dist) < 1.2 * r

    return np.array(idx)[p]

def weight(density, mesh):

    return mesh.volume * density

print(getABQFromStpFile('models/FFUCA finished 20220713.stp'))

# if __name__ == "__main__":
#     print(getABQFromStpFile('models/E10488316.SNU001+A+CONTROL ARM HOUSING.stp'))