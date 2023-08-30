

import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.neighbors import NearestNeighbors


model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))



def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features




def get_file_list(root_dir):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list



def train(filenames):
    feature_list = []
    for i in tqdm(range(len(filenames))):
        feature_list.append(extract_features(filenames[i], model))

    pickle.dump(feature_list, open('/Users/prakhargupta/reverseImageSearch/Reverse_Image_Search.ipynbfeatures-caltech101-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('/Users/prakhargupta/reverseImageSearch/Reverse_Image_Search.ipynbfilenames-caltech101.pickle','wb'))

   


def search(testFeature):
    
    filenames = pickle.load(open('/Users/prakhargupta/reverseImageSearch/Reverse_Image_Search.ipynbfilenames-caltech101.pickle', 'rb'))
    feature_list = pickle.load(open('/Users/prakhargupta/reverseImageSearch/Reverse_Image_Search.ipynbfeatures-caltech101-resnet.pickle', 'rb'))
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(feature_list)
    distances, indices = neighbors.kneighbors([testFeature])
    print(len(feature_list[0]))

    # plt.imshow(mpimg.imread(testFeature))
    plt.imshow(mpimg.imread(filenames[indices[0][2]]))
    plt.show()



root_dir = '/Users/prakhargupta/Downloads/caltech101'
filenames = sorted(get_file_list(root_dir))
# plt.imshow(mpimg.imread(filenames[1340]))
# train(filenames)

testFeature = extract_features("/Users/prakhargupta/Downloads/testplaneimage.jpeg",model)
# print(len(testFeature))
search(testFeature)
