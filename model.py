import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        X_test=[]
        Y_test=[]
        X_train=[]
        Y_train=[]
        for item in test_data:
            X_test.append(np.array(item[0].reshape(1,-1)).flatten())
            Y_test.append(item[1])
        self.x_test = X_test
        self.y_test = Y_test

        for item in train_data:
            X_train.append(np.array(item[0].reshape(1,-1)).flatten())
            Y_train.append(item[1])
        self.x_train = X_train
        self.y_train = Y_train
        # print(self.x_train)
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        if model_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'RF':
            clf = RandomForestClassifier(n_estimators=150, random_state=42)
        elif model_name == 'AB':
            clf = AdaBoostClassifier(n_estimators=200, random_state=42)
        return clf
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        self.model.fit(self.x_train,self.y_train)
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(self.y_test,y_pred))
    
    def classify(self, input):
        return self.model.predict(input)[0]
        

