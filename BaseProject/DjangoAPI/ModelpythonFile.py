

import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from annoy import AnnoyIndex


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.neighbors import NearestNeighbors
dirname = os.path.dirname(__file__)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3),pooling ='avg')


def extract_features(img_path):
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
    feature_list = pickle.load(open('/home/f20200697/features.pickle', 'rb'))
    filenames = pickle.load(open(os.path.join(dirname, 'Reverse_Image_Search_s3.ipynbfilenames-caltech101.pickle'), 'rb'))
    # neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(feature_list)
    # distances, indices = neighbors.kneighbors([testFeature])
    # print(len(feature_list[0]))
    # plt.imshow(mpimg.imread(testFeature))
    # plt.imshow(mpimg.imread(filenames[indices[0][0]]))
    # return indices


    annoy_index = AnnoyIndex(2048)  # Length of item vector that will be
    num_items = len(feature_list)
    for i in range(num_items):
        annoy_index.add_item(i, feature_list[i])
    annoy_index.build(40)
    res = annoy_index.get_nns_by_vector(feature_list[4], 5, include_distances=True)
    indices = res[0]
    distances = res[1]
    responseData = []
    for i in range( len(indices[0])):
    #     print(i)
        responseData.append(filenames[indices[0][i]])
    print(indices)
    plt.show()
    return responseData,distances


# root_dir = '/Users/prakhargupta/Downloads/caltech101'
# filenames = sorted(get_file_list(root_dir))
# plt.imshow(mpimg.imread(filenames[1340]))
# train(filenames)

# testFeature = extract_features("/Users/prakhargupta/Downloads/testplaneimage.jpeg")
# print(len(testFeature))
# search(testFeature)
