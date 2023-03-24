import numpy as np
import pyransac3d as pyrsc
import trimesh

def getCylinderNodes(vertices,nodeTags,dist=45):
    '''Enumarate idx sets of faces (n,3) _i=inner hardpoint _o 0 outer hardpoint'''

    faces=[[i,i+1,i+2] for i in range(0,len(vertices),3)]

    'sort array of ascending vertices along longest axis (Y) '
    p=vertices[:, 1].argsort()

    'get limits of longest axis (Y)'
    yMax=max(vertices[:,1])
    yMin=min(vertices[:,1])

    'Create sample subsets of both ends along longest extension (Y)'
    vert_o=vertices[vertices[:, 1] < yMin+dist][::100]
    vert_i=vertices[vertices[:, 1] > yMax-dist][::100]

    print(yMax,yMin)

    'Create Fitting Instances'
    sph = pyrsc.Sphere()
    cyl = pyrsc.Cylinder()

    'Find Sphere Parameter (center, radii and inliner) of both sample-subsets (vert_o and vert_i) '
    c_o,r,inliners = sph.fit(vert_o, thresh=1.0, maxIteration=1000)
    c_i,r,inliners = sph.fit(vert_i, thresh=1.0, maxIteration=1000)

    'Create Trimesh mesh'
    mesh=trimesh.Trimesh(vertices=vertices, faces=faces)

    'Measure real Bore Radii'
    r_o = mesh.nearest.on_surface([c_o])[1][0]
    r_i = mesh.nearest.on_surface([c_i])[1][0]

    'Create Fitting sample subsets of the bore'
    inliners_o=fetchCylPts(vertices[p], r_o, c_o)
    inliners_i=fetchCylPts(vertices[p], r_i, c_i)

    'Create Vertices sets of bore'
    v_o=vertices[p][inliners_o]
    v_i=vertices[p][inliners_i]

    'Find Cylinder Parameter of both subsets (v_o and v_i)'
    c_o, axis_o, r_o, inliners_o = cyl.fit(v_o, thresh=1.0, maxIteration=1000)
    c_i, axis_i, r_i, inliners_i = cyl.fit(v_i, thresh=1.0, maxIteration=1000)

    'Update refined Vertices sets of bore after Cylinder Fitting'
    v_o=v_o[inliners_o]
    v_i=v_i[inliners_i]

    'Match cylinder fitted sets with nodeTags'
    nT_o=nodeTags[p][inliners_o]
    nT_i=nodeTags[p][inliners_i]

    'Visualizations with Trimesh'
    c=[c_o,c_i]

    print(sum([max(v_i[:, 0]), min(v_i[:, 0])]))
    print(sum([max(v_i[:, 1]), min(v_i[:, 1])]))
    print(sum([max(v_i[:, 2]), min(v_i[:, 2])]))
    print(np.roll(axis_i,2))
    print(getlinePlaneIntersec(np.roll(np.around(axis_i),2),np.array(c_i),np.array(axis_o),np.array(c_o)))

    kPt=trimesh.points.PointCloud(c, colors=([1,0,0,0.9],[0,0,1,0.9]))
    pcl_o=trimesh.points.PointCloud(v_o,np.tile(np.array([1, 0, 0, 0.9]), (len(v_o), 1)))
    pcl_i=trimesh.points.PointCloud(v_i,np.tile(np.array([0, 0, 1, 0.9]), (len(v_i), 1)))

    scene = trimesh.Scene([kPt,mesh,pcl_o,pcl_i])

    #scene.show()

    return nT_o, nT_i

def fetchCylPts(pts, r, c, band=1.15):
    dist = [np.linalg.norm(pts[i] - c) for i in range(0, len(pts))]
    idx = [i for i in range(0, len(pts))]
    p = np.array(dist) < band* r

    return np.array(idx)[p]

def getlinePlaneIntersec(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint

    return Psi

if __name__ == "__main__":
    getCylinderNodes()