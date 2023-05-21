import numpy as np
from sklearn.preprocessing import LabelEncoder

def FillingData(data):
    #print("NULL= ",data['gender'].isna().sum())
    #print("The number of males in the first type = ",data[(data["species"] == 'Adelie') & (data["gender"] == 'male')]['gender'].count())
    #print("The number of females in the first type = ",data[(data["species"] == 'Adelie') & (data["gender"] == 'female')]['gender'].count())

    #print("The number of males in the Second type = ",data[(data["species"] == 'Gentoo') & (data["gender"] == 'male')]['gender'].count())
    #print("The number of females in the Second type = ",data[(data["species"] == 'Gentoo') & (data["gender"] == 'female')]['gender'].count())

    #print("The number of males in the Third type = ",data[(data["species"] == 'Chinstrap') & (data["gender"] == 'male')]['gender'].count())
    #print("The number of female in the Third type = ",data[(data["species"] == 'Chinstrap') & (data["gender"] == 'female')]['gender'].count())

    data[0:50]['gender'].replace([np.NaN], "female", inplace=True)
    data[50:100]['gender'].replace([np.NaN], "male", inplace=True)
    #print("NULL= ",data['gender'].isna().sum())

    return data

def Feature_Encoder(data):
    lbl = LabelEncoder()
    lbl.fit(list(data['gender'].values))
    data['gender'] = lbl.transform(list(data['gender'].values))
    return data


# Scaling (Try after get acc)
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X




