'''
Final JSModel

requires following 6 files 
to be in the working dir
to successfully run:

- classification_tfidf_model.onxx
- clustering_tfidf_model.onxx
- classification_model.onxx
- clustering_model.onxx
- classification_features.json
- clustering_features.json
'''


import json
import re
import os
import numpy as np
import pandas as pd 

import onnxruntime as rt

classification_labels = {
"0": "marketing",
"1": "cdn",
"2": "tag-manager",
"3": "video",
"4": "customer-success",
"5": "utility",
"6": "ads",
"7": "analytics",
"8": "hosting",
"9": "content",
"10": "social",
"11": "other"
}

clustering_labels = {
"1": "noncritical",
"0": "critical"
}

class JSModel_ONNX(object):
    def __init__(self):
# classification_test.onnx
        self.sess_classification = rt.InferenceSession("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models/classification_test.onnx")
        self.sess_clustering = rt.InferenceSession("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models//clustering_test.onnx")

        self.sess_classification_tfidf = rt.InferenceSession("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models/classification_tfidf_test.onnx")
        self.sess_clustering_tfidf = rt.InferenceSession("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models/clustering_tfidf_test.onnx")

        with open("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/classification_features.json") as f:
            self.classification_features = json.loads(f.read())["features"]

        with open("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/clustering_features.json") as f:
            self.clustering_features = json.loads(f.read())["features"]
    
    
        self.classification_kws = []
        for feature in self.classification_features:
            tmp = feature.split("|")
            self.classification_kws += tmp
        self.classification_kws = list(set(self.classification_kws))
        print(self.classification_kws)

        self.clustering_kws = []
        for feature in self.clustering_features:
            tmp = feature.split("|")
            self.clustering_kws += tmp
        self.clustering_kws = list(set(self.clustering_kws))

    def get_scripts_features(self, data, kws, features):
        resultant_features = []
        scripts_kws = []
        for kw in kws:
            scripts_kws += [kw]*data.count("."+kw+"(")
            scripts_kws += [kw]*data.count("."+kw+" (")
        for ft in features:
            if "|" not in ft:
                resultant_features += [ft]*scripts_kws.count(ft)
            else:
                singular_kws = ft.split("|")
                if len([ele for ele in singular_kws if ele in scripts_kws]) == len(singular_kws):
                    resultant_features += [ft]
        return resultant_features

    def get_scripts_classification_features(self, data):
        return " ".join(self.get_scripts_features(data, self.classification_kws, self.classification_features))

    def get_scripts_clustering_features(self, data):
        return " ".join(self.get_scripts_features(data, self.clustering_kws, self.clustering_features))

    def printt(self):
        print(self.sess_classification_tfidf)
        print(self.sess_clustering_tfidf)
        print(self.sess_classification)
        print(self.sess_clustering)

    def predict(self, script):
        
        script = re.sub("\s+", " ", script)
        reduced_script = self.get_scripts_classification_features(script)
        tfidf_representation = self.sess_classification_tfidf.run(None, {"float_input": np.array([reduced_script]).reshape(1,1)})[0]
        prediction = self.sess_classification.run(None, {'float_input': tfidf_representation.astype(np.float32)})[0] 
        
        # Max Probability
        onx_pred = self.sess_classification.run(None, {'float_input': tfidf_representation.astype(np.float32)})[1] 
        df = pd.DataFrame(onx_pred)

        if df.values.max() > 0.8:
            return classification_labels[str(prediction[0])] 
        else:
            reduced_script = self.get_scripts_clustering_features(script)
            tfidf_representation = self.sess_clustering_tfidf.run(None, {"float_input": np.array([reduced_script]).reshape(1,1)})[0]
            prediction = self.sess_clustering.run(None, {'float_input': tfidf_representation.astype(np.float32)})[0]   
            return clustering_labels[str(prediction[0])]

    
    def compare_classification(self, script):
        
        script = re.sub("\s+", " ", script)
        reduced_script = self.get_scripts_classification_features(script)
        tfidf_representation = self.sess_classification_tfidf.run(None, {"float_input": np.array([reduced_script]).reshape(1,1)})[0]
        prediction = self.sess_classification.run(None, {'float_input': tfidf_representation.astype(np.float32)})[0] 
        
        # Max Probability
        onx_pred = self.sess_classification.run(None, {'float_input': tfidf_representation.astype(np.float32)})[1] 
        df = pd.DataFrame(onx_pred)
        
       
        return df.values
    
    def compare_clustering(self, script):
        
        script = re.sub("\s+", " ", script)
        reduced_script = self.get_scripts_classification_features(script)
        
        tfidf_representation = self.sess_clustering_tfidf.run(None, {"float_input": np.array([reduced_script]).reshape(1,1)})[0]
        prediction = self.sess_clustering.run(None, {'float_input': tfidf_representation.astype(np.float32)})[0]   
        return clustering_labels[str(prediction[0])]
        

