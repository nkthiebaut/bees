__author__ = 'thiebaut'

import os

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


FTRAIN = './data/images/kaggle-facial-keypoint-detection/training.csv'
FTEST = './data/images/kaggle-facial-keypoint-detection/test.csv'

# load the labels using pandas
labels = pd.read_csv("./data/train_labels.csv",
                     index_col=0)

submission_format = pd.read_csv("data/SubmissionFormat.csv",
                                index_col=0)

def get_image(row_or_str, root="data/images/"):
    # if we have an instance from the data frame, pull out the id
    # otherwise, it is a string for the image id
    if isinstance(row_or_str, pd.core.series.Series):
        row_id = row_or_str.name
    else:
        row_id = row_or_str

    filename = "{}.jpg".format(row_id)

    # test both of these so we don't have to specify. Image should be
    # in one of these two. If not, we let Image.open raise an exception.
    train_path = os.path.join(root, "train", filename)
    test_path = os.path.join(root, "test", filename)

    file_path = train_path if os.path.exists(train_path) else test_path

    return np.array(Image.open(file_path), dtype=np.int32)

def create_feature_matrix(label_dataframe):
    n_imgs = label_dataframe.shape[0]

    # initialized after first call to
    feature_matrix = None

    for i, img_id in tqdm(enumerate(label_dataframe.index)):
        features = preprocess(get_image(img_id))

        # initialize the results matrix if we need to
        # this is so n_features can change as preprocess changes
        if feature_matrix is None:
            n_features = features.shape[0]
            feature_matrix = np.zeros((n_imgs, n_features), dtype=np.float32)

        if not features.shape[0] == n_features:
            print "Error on image {}".format(img_id)
            features = features[:n_features]

        feature_matrix[i, :] = features

    return feature_matrix

# turn those images into features!
bees_features = create_feature_matrix(labels)

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 3, 200, 200)
    return X, y


X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))