import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    truepositive = 0
    falseneg = 0
    falsepos = 0

    for index in range(len(predicted_labels)):
        if predicted_labels[index] == 1 and real_labels[index] == 1:
            truepositive += 1
        if predicted_labels[index] == 0 and real_labels[index] == 1:
            falseneg += 1
        if predicted_labels[index] == 1 and real_labels[index] == 0:
            falsepos += 1

    if truepositive + falseneg == 0 or truepositive + falsepos == 0:
        return 0
    
    precision = truepositive / (truepositive + falsepos)
    recall = truepositive / (truepositive + falseneg)

    if precision + recall == 0:
        return 0

    fscore = 2 * (precision * recall) / (precision + recall)
    return fscore


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p = 3 
        summation = 0.0
        for i in range(len(point1)):
            difference = abs(point1[i] - point2[i])
            summation += difference ** p
        minkowd = summation ** (1 / p)
        return minkowd

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        summation = 0.0
        for i in range(len(point1)):
            difference = point1[i] - point2[i]
            summation += difference ** 2
        edist = summation ** 0.5
        #print(edist)
        return edist

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dotprod = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for i in range(len(point1)):
           dotprod += point1[i] * point2[i]
           norm1 += point1[i] ** 2
           norm2 += point2[i] ** 2
        if norm1 == 0 or norm2 == 0:
           return 1.0 
        cosinesim = dotprod / ((norm1 ** 0.5) * (norm2 ** 0.5))
        cdist = 1 - cosinesim
        return cdist



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        k = 1
        bestF = -1
        #worst case bestFinfo. cannot be initialized as empty
        bestFinfo = [0,'cosine_dist',29]
        funcval = {'euclidean':0,'minkowski':1,'cosine_dist':2}

        while k < 30:
            for func in distance_funcs:
                KNNnoscale = KNN(k,distance_funcs[func])
                KNNnoscale.train(x_train,y_train)
                predicted_labels = KNNnoscale.predict(x_val)
                fscore = f1_score(y_val,predicted_labels)
                if bestF == fscore:
                    #if the f scores are equal and new func value is less than bestf score, then replace best F info.
                    if funcval.get(bestFinfo[1]) > funcval.get(func):
                        bestFinfo = [fscore,func,k,KNNnoscale]
                    #if functions are equal, we don't need to compare k as all future "k"'s will be larger. 
                    #for points with smaller k's, there would be a higher ranking tiebreaking criteria that would take priority.
                if bestF < fscore:
                    bestFinfo = [fscore,func,k,KNNnoscale]
                    bestF= fscore
            k+=2
        self.best_k = bestFinfo[2]
        self.best_distance_function = bestFinfo[1]
        self.best_model = bestFinfo[3]

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        bestF = -1
        #worst case bestFinfo. cannot be initialized as empty
        bestFinfo = [0,'cosine_dist',29,'normalize']
        funcval = {'euclidean':0,'minkowski':1,'cosine_dist':2}
        scaledict = {'min_max_scale': 0,'normalize': 1,}
        for scalar_name in scaling_classes:
            scaler = scaling_classes[scalar_name]()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            k = 1
            while k < 30:
                for func in distance_funcs:
                    KNNscaled = KNN(k,distance_funcs[func])
                    KNNscaled.train(x_train_scaled,y_train)
                    predicted_labels = KNNscaled.predict(x_val_scaled)
                    fscore = f1_score(y_val,predicted_labels)
                    if bestF == fscore:
                        # if the scaler is minmax and bestF is normalize, then replace bestFinfo
                        if scaledict.get(scalar_name) < scaledict.get(bestFinfo[3]): 
                            bestFinfo = [fscore, func, k, scalar_name, KNNscaled]
                        elif scaledict.get(scalar_name) == scaledict.get(bestFinfo[3]):
                            # If scalers are the same, prefer the lower function values, and replace bestFinfo
                            if funcval.get(func) < funcval.get(bestFinfo[1]):
                                bestFinfo = [fscore, func, k, scalar_name, KNNscaled]
                    if bestF < fscore:
                        bestFinfo = [fscore,func,k,scalar_name,KNNscaled]
                        bestF= fscore
                k+=2
        self.best_k = bestFinfo[2]
        self.best_distance_function = bestFinfo[1]
        self.best_scaler = bestFinfo[3]
        self.best_model = bestFinfo[4]


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized = []
        normx = 0
        for row in features:
            sum = 0
            #taking the sum of squares in the list
            for i in range(len(row)):
                sum += row[i]**2
            normx = sum**0.5
            newrow = []
            #divide all values in that list by its norm
            for j in range(len(row)):
                #to avoid dividing by 0
                if normx == 0:
                    newrow.append(0)
                else:
                    newrow.append(row[j]/normx)
            normalized.append(newrow)
        return normalized


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        minmaxlist = []
        normalized = 0
        totalmaxminvals = []
        #get min and max per column a.k.a (row[ith])
        for i in range(len(features[0])):
            transformedlist = []
            for j in range(len(features)):
                transformedlist.append(features[j][i])
            
            maxval = max(transformedlist)
            minval = min(transformedlist)
            #put values into sublists
            totalmaxminvals.append([maxval,minval])
        #apply normalization
        for row in features:
            normvals= []
            for i in range(len(row)):
                if totalmaxminvals[i][0]-totalmaxminvals[i][1] == 0:
                    normalized = 0
                else:
                    normalized = (row[i]-totalmaxminvals[i][1])/(totalmaxminvals[i][0]-totalmaxminvals[i][1])
                normvals.append(normalized)
            minmaxlist.append(normvals)
        return minmaxlist
