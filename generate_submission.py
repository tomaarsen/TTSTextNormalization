
import gc, os, sys, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

class Constants:
    # Class storing some parameters
    limit = 5750000
    start = 5
    end = 0
    chars = start + end
    words = 4
    features = 5

    version = "15"

    prefix = os.path.dirname(__file__)
    # Location of file to use for training
    train_data_csv = os.path.join(prefix, "data/raw/en_test_2.csv")
    # Location of file to store preprocessed data
    classify_train_file = os.path.join(prefix, f"data/custom/en_test2_{chars}c_{words}w_{features}f.npz")
    labels = ["PLAIN", "PUNCT", "DATE", "LETTERS", "CARDINAL", "VERBATIM", "DECIMAL", "MEASURE", "MONEY", "ORDINAL", "TIME", "ELECTRONIC", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"]

class Data(object):

    @staticmethod
    def _get_file_data():
        # Load data from the preprocessed file

        # https://stackoverflow.com/a/56243777
        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        temp = np.load(Constants.classify_train_file)
        # Get tuple with the entire dataframe, as well as the the x array
        out = (pd.DataFrame(temp['train'], columns=["sentence_id", "token_id", "before"]), temp['x_train'])
        np.load = np_load_old
        return out

    @staticmethod
    def _load_data():
        # Load data from the training dataset directly
        return pd.read_csv(Constants.train_data_csv)

    @staticmethod
    def get_data_limit(limit):
        # Get the first `limit` training values. 
        # Returns a tuple of the training dataframe and the x array 

        # Either use preprocessed, stored data
        if os.path.exists(Constants.classify_train_file):
            train, x_train = Data._get_file_data()
            return train[:limit], x_train[:limit]
        
        # Or load data, and store the preprocessed version if the limit is 
        # larger than the size of the full training set
        train = Data._load_data()
        save = False
        if train.shape[0] < limit:
            save = True
        train = train[:limit]
        return Data._get_data(train, save=save)

    @staticmethod
    def get_data_limit_from(limit):
        # Get all training values after the entry indexed by `limit`. 
        # Returns a tuple of the training dataframe, the x array and the y array.

        # Either use preprocessed, stored data
        if os.path.exists(Constants.classify_train_file):
            train, x_train = Data._get_file_data()
            return train[limit:], x_train[limit:]

        # Or load data from the raw training csv and preprocess it
        train = Data._load_data()
        train = train[limit:]
        return Data._get_data(train, save=False)

    @staticmethod
    def get_data_percentage(percentage):
        # Use the preprocessed data if the percentage ratio is 1, and the file exists
        if os.path.exists(Constants.classify_train_file) and percentage == 1:
            return Data._get_file_data()

        # Load data, partition the sentences, and preprocess it
        # also store the data if the percentage is 1
        train = Data._load_data()
        train = train[train["sentence_id"] % round(1/percentage) == 0]
        return Data._get_data(train, save=percentage == 1)

    @staticmethod
    def _get_data(train, save=False):
        # Ensure all tokens are strings
        train['before'] = train['before'].astype(np.str)
        label_dict = {label: i for i, label in enumerate(Constants.labels)}
        # Each sentence in the training data is passed to the `convert_data` function, which preprocesses it.
        # All preprocessed entries are then concatenated into x_train        
        x_train = np.array(train[["before", "sentence_id"]].groupby("sentence_id")["before"].apply(convert_data).explode().reset_index(drop=True).tolist())
        # Save the training dataframe, as well as x_train to prevent having to preprocess every time
        if save:
            print("Saving data to csv")
            np.savez(Constants.classify_train_file, train=train, x_train=x_train)
        else:
            print("Not saving data to csv")
        print("Fetched data")
        return train, x_train

def convert_data(series):
    
    # Get values from Constants
    start = Constants.start
    end = Constants.end
    # Number of features per word is start + end
    num_features = start + end
    # Prepare arrays for use
    x_train = np.zeros((series.shape[0], num_features))
    metainfo = np.zeros((series.shape[0], 5))
    # Per word, add the first `start` characters
    for w_index, word in enumerate(series):
        for c_index, c in enumerate(word[:start]):
            x_train[w_index, c_index] = ord(c)

        # Store metainformation for each word
        # Index 0: # of digits
        # Index 1: # of uppercase
        # Index 2: # of non alphanumerical characters
        # Index 3: Length of word
        # Index 4: # of vowels
        vowels = ['a','e','i','o','u']
        metainfo[w_index, 3] = len(word)
        for c in word:
            if c.isdigit():
                metainfo[w_index, 0] += 1
            if c.isupper():
                metainfo[w_index, 1] += 1
            if not c.isalnum():
                metainfo[w_index, 2] += 1
            if c in vowels:
                metainfo[w_index, 4] += 1

    # Pad the x_train with empty values before and after
    x_train = np.lib.pad(x_train, ((2, 1), (0, 0)), "constant", constant_values=(0,))
    # Convert x_train to a concatenation of x_train offset 4 times, and the metainfo
    # The result is that x_train is "previous word" + "this word" + "next word" + "second next word" + "metainfo"
    x_train = np.concatenate((x_train[:-3], x_train[1:-2], x_train[2:-1], x_train[3:], metainfo),axis=1)
    # Convert to np.int8 for memory use
    return x_train.astype(np.int8)

class Utility(object):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def predict(model):
        # Get the training dataframe and the X array to predict on
        train, x_train = Data.get_data_percentage(1)
        # Get a DMatrix used for prediction
        dtrain = xgb.DMatrix(x_train)
        pred = model.predict(dtrain)
        # Add the predicted class to the dataframe and return it
        train["token_class"] = pred
        return train

    @staticmethod
    def convert(df):
        # Add an after column with converted tokens

        from converters.Plain      import Plain
        from converters.Punct      import Punct
        from converters.Date       import Date
        from converters.Letters    import Letters
        from converters.Cardinal   import Cardinal
        from converters.Verbatim   import Verbatim
        from converters.Decimal    import Decimal
        from converters.Measure    import Measure
        from converters.Money      import Money
        from converters.Ordinal    import Ordinal
        from converters.Time       import Time
        from converters.Electronic import Electronic
        from converters.Digit      import Digit
        from converters.Fraction   import Fraction
        from converters.Telephone  import Telephone
        from converters.Address    import Address

        labels = ["PLAIN", "PUNCT", "DATE", "LETTERS", "CARDINAL", "VERBATIM", "DECIMAL", "MEASURE", "MONEY", "ORDINAL", "TIME", "ELECTRONIC", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"]
        label_dict = {
            "PLAIN": Plain(),
            "PUNCT": Punct(),
            "DATE": Date(),
            "LETTERS": Letters(),
            "CARDINAL": Cardinal(),
            "VERBATIM": Verbatim(),
            "DECIMAL": Decimal(),
            "MEASURE": Measure(),
            "MONEY": Money(),
            "ORDINAL": Ordinal(),
            "TIME": Time(),
            "ELECTRONIC": Electronic(),
            "DIGIT": Digit(),
            "FRACTION": Fraction(),
            "TELEPHONE": Telephone(),
            "ADDRESS": Address(),
        }

        df["after"] = df.apply(lambda b: label_dict[labels[int(b["token_class"])]].convert(b.before), axis=1)
        import pdb; pdb.set_trace()
        return df

if __name__ == "__main__":
    # Get the name of the file path to load the model from
    #xgb_model = os.path.join(Constants.prefix, f"models/xgb_sub{round(Constants.limit/1000)}_{Constants.chars}c_{Constants.words}w_{Constants.features}f_v{Constants.version}_model.dat")
    xgb_model = os.path.join(Constants.prefix, "best/xgb_sub5750_5c_4w_5f_v1_model.dat")
    
    # Load the model
    model = pickle.load(open(xgb_model, "rb"))

    # Predict classes using the loaded model    
    df = Utility.predict(model)
    import pdb; pdb.set_trace()
    # Apply conversion on all tokens
    Utility.convert(df)

    # Create a dataframe using the requested format
    #new_df = df[["sentence_id", "token_id", "after"]]
    #new_df["id"] = new_df.apply(lambda x: str(x["sentence_id"]) + "_" + str(x["token_id"]), axis=1)
    #new_df = new_df[["id", "after"]]
    # Save the submission
    #new_df.to_csv("submission.csv", index=False)
