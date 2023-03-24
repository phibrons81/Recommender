import numpy as np
import math
from scipy.stats import moment
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


class CLassifier():

    def getObjectFeatureDf(self, mesh):
        # Returns a pandas df with all Moments of a given stl-File
        # stlPath = Path to input-File
        # normalizedAccM200: defines, wheter moments are normalized acc. to M200; true = normalized, False= not Normalized

        points_3d = mesh.sample_points_uniformly(number_of_points=20000)

        points = np.asarray(points_3d.points)  # Sampled by Open3D

        normalizedAccM200 = True

        def getMoment(M, points):
            M_step1 = np.power(points, M)
            M_step2 = np.prod(M_step1, axis=1)
            M_hat = sum(M_step2) / len(points)
            return M_hat

        COM = moment(points, 1)

        points_centered = np.subtract(points, COM)

        S = np.asarray([getMoment([2, 0, 0], points_centered), getMoment([1, 1, 0], points_centered),
                        getMoment([1, 0, 1], points_centered), \
                        getMoment([1, 1, 0], points_centered), getMoment([0, 2, 0], points_centered),
                        getMoment([0, 1, 1], points_centered), \
                        getMoment([1, 0, 1], points_centered), getMoment([0, 1, 1], points_centered),
                        getMoment([0, 0, 2], points_centered)])
        S = np.reshape(S, [3, 3])

        U, S1, VH = np.linalg.svd(S, full_matrices=True)  # Singular Value Decomposition

        if normalizedAccM200 == True:
            U_Scale = np.divide(U, math.sqrt(S1[0]))
        else:
            U_Scale = U

        points_scale_rot = []
        for i in points_centered:
            points_scale_rot.append(np.dot(i, U_Scale))
        points_scale_rot = np.reshape(points_scale_rot, [len(points_centered), 3])

        # Calculation of Feature-Vector
        featureVectorObject = np.asarray([getMoment([2, 0, 0], points_scale_rot),
                                          getMoment([0, 2, 0], points_scale_rot),
                                          getMoment([0, 0, 2], points_scale_rot),

                                          getMoment([0, 2, 2], points_scale_rot),
                                          getMoment([2, 0, 2], points_scale_rot),
                                          getMoment([2, 2, 0], points_scale_rot),

                                          getMoment([3, 0, 0], points_scale_rot),
                                          getMoment([0, 3, 0], points_scale_rot),
                                          getMoment([0, 0, 3], points_scale_rot),

                                          getMoment([0, 3, 3], points_scale_rot),
                                          getMoment([3, 0, 3], points_scale_rot),
                                          getMoment([3, 3, 0], points_scale_rot),

                                          getMoment([4, 0, 0], points_scale_rot),
                                          getMoment([0, 4, 0], points_scale_rot),
                                          getMoment([0, 0, 4], points_scale_rot)])

        self.featureVector = np.asarray(featureVectorObject)

        assignments = {"FileName": "InputModel",
                       "M200": self.featureVector[0],
                       "M020": self.featureVector[1],
                       "M002": self.featureVector[2],

                       "M022": self.featureVector[3],
                       "M202": self.featureVector[4],
                       "M220": self.featureVector[5],

                       "M300": self.featureVector[6],
                       "M030": self.featureVector[7],
                       "M003": self.featureVector[8],

                       "M033": self.featureVector[9],
                       "M303": self.featureVector[10],
                       "M330": self.featureVector[11],

                       "M400": self.featureVector[12],
                       "M040": self.featureVector[13],
                       "M004": self.featureVector[14]}
        
        #print(pd.DataFrame(assignments, index=[0]))

        df = pd.DataFrame(assignments, index=[0])

        return df,self.featureVector

    def getResultDf(self,queryDf, nResults, enoviadataPath,armTypeQuery='2 Pt. Guiding Control Arm'):
        #Returns a pandas DF with the nearest data for a given feature-vector
        #reccomder = sklearn NearestNeighbors-Object (already fitted !)

        #armTypeQuery = "3 Pt. Guiding Control Arm (U-Shape)"
        dfAllArms = pd.read_csv(enoviadataPath, delimiter=",", index_col=0)
        if armTypeQuery !='all':

            dfAllArms =  dfAllArms[dfAllArms["Type"] == armTypeQuery]

        #Reindex
        dfAllArms.reset_index(inplace=True)

        featureNames = ["M020","M022","M202","M220","M030","M003","M033","M400","M040","M004"]

        features = dfAllArms[featureNames].values
        objFeatures = queryDf[featureNames]
        #objFeatures = objFeatures.as_matrix()
        objFeatures = objFeatures.values

        #Normalisation of all cols(Features): Append Object Feature, Normalize, delte Obj Feature
        features = np.vstack((features,objFeatures))
        scaler = MinMaxScaler()
        scaler.fit(features)
        features = scaler.transform(features)

        objFeatures = features[len(features)-1,:]
        features = features[0:len(features)-2,:]

        #Generate Recommender-Object
        recommender = NearestNeighbors(n_neighbors=nResults)
        recommender.fit(features)

        objFeatures = objFeatures.reshape(1, -1) #Required by sklearn

        res = recommender.kneighbors(objFeatures) #Returns [Distances,Indices]
        #print(res)
        indices = res[1].tolist() #Returns [[indices]] of reults
        indices = indices[0]
        distances = res[0].tolist() #Returns [[distances]] of reults
        distances = distances[0]
        result = dfAllArms.loc[indices] #Get complete Data of all results
        result["Distance"] = distances #Add Col for distances to each result

        return result