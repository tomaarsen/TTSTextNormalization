
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
    epoch = 200

    version = "18"

    prefix = os.path.dirname(__file__)
    # Location of file to use for training
    train_data_csv = os.path.join(prefix, "data/raw/en_train.csv")
    # Location of file to store preprocessed data
    classify_train_file = os.path.join(prefix, f"data/custom/en_train_{chars}c_{words}w_{features}f.npz")
    labels = ["PLAIN", "PUNCT", "DATE", "LETTERS", "CARDINAL", "VERBATIM", "DECIMAL", "MEASURE", "MONEY", "ORDINAL", "TIME", "ELECTRONIC", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"]
    # Parameters used in training
    params = {
        'eta': 0.3, # Eg learning rate, 0.3 default
        'gamma': 0, # Eg Min-split-loss, 0 default
        # 4 is worse. 12 is slightly worse.
        'max_depth':10, # More likely to overfit at higher values, 6 default
        'min_child_weight': 1, # Higher = more conservative, 1 default
        'min_delta_step': 0, # If 0, no constraint. can help make update step more conservative. 0 default
        'subsample': 1, # Get section of traning instances. 1 default
        'objective':'multi:softmax',
        'num_class':len(labels),
        'eval_metric':'merror',
        'colsample_bytree': 1,
        'silent':1,
        'seed':0,
        'gpu_id': 0, # Custom
        'tree_method': 'gpu_hist', # Custom
    }

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
        # Get tuple with the entire dataframe, as well as the the x and the y arrays
        out = (pd.DataFrame(temp['train'], columns=["sentence_id", "token_id", "class", "before", "after"]), temp['x_train'], temp['y_train'])
        np.load = np_load_old
        return out

    @staticmethod
    def _load_data():
        # Load data from the training dataset directly
        return pd.read_csv(Constants.train_data_csv)

    @staticmethod
    def get_data_limit(limit):
        # Get the first `limit` training values. 
        # Returns a tuple of the training dataframe, the x array and the y array.

        # Either use preprocessed, stored data
        if os.path.exists(Constants.classify_train_file):
            train, x_train, y_train = Data._get_file_data()
            return train[:limit], x_train[:limit], y_train[:limit]
        
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
            train, x_train, y_train = Data._get_file_data()
            return train[limit:], x_train[limit:], y_train[limit:]
        
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
        # y_train is the class of train converted to some integer between 0 and 15, including edges
        y_train = np.array(train["class"].apply(lambda x: label_dict[x])).astype(np.int8)
        # Each sentence in the training data is passed to the `convert_data` function, which preprocesses it.
        # All preprocessed entries are then concatenated into x_train
        x_train = np.array(train[["before", "sentence_id"]].groupby("sentence_id")["before"].apply(convert_data).explode().reset_index(drop=True).tolist())
        # Save the training dataframe, as well as x_train and y_train to prevent having to preprocess every time
        if save:
            print("Saving data to csv")
            np.savez(Constants.classify_train_file, train=train, x_train=x_train, y_train=y_train)
        else:
            print("Not saving data to csv")
        print("Fetched data")
        return train, x_train, y_train

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

class Train(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def train_percentage(percentage):
        # Train using `percentage` of all data
        train, x_train, y_train = Data.get_data_percentage(percentage)
        return Train._train(x_train, y_train)

    @staticmethod
    def train_limit(limit):
        # Train using first `limit` values of all data
        train, x_train, y_train = Data.get_data_limit(limit)
        return Train._train(x_train, y_train)

    @staticmethod
    def _train(x_train, y_train):
        # Train a model
        # Split X and y 90-10, such that 90% is the training data and 10% is the testing data
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
        # Construct DMatrices to use in training. 
        # One is used for testing, and one for keeping track of the performance
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(d_test, 'test'), (d_train, 'train')]
        
        gc.collect()
        # Train a model using the parameters in Constants
        model = xgb.train(Constants.params, d_train, Constants.epoch, watchlist, early_stopping_rounds=5, verbose_eval=10)
        return model

class Test(object):
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def test(model, limit=False):
        # Get the x and y from data, either using a limit or with 100% of the data
        if limit:
            train, x_test, y_test = Data.get_data_limit_from(Constants.limit)
        else:
            train, x_test, y_test = Data.get_data_percentage(1)
        # Create a DMatrix to use in prediction
        d_test = xgb.DMatrix(x_test, label=y_test)
        # Predict classes using the model
        pred = model.predict(d_test)
        # Create an output dict
        out = pd.DataFrame({"Before": train["before"], "After": train["after"], "Predicted": pred, "Correct": y_test})
        # Save all errors
        #out[out["Predicted"] != out["Correct"]].to_csv(f"errors.csv", index=False)
        # Save all data with the predictions
        #out.to_csv(f"out.csv", index=False)
        print(f"{100 * (1 - out[out['Predicted'] != out['Correct']].shape[0] / out.shape[0]):.8f}% accuracy on the", ("entire dataset" if not limit else "remainder of the dataset"))
        return out

    @staticmethod
    def convert(df):

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

        df["cust_after"] = df.apply(lambda b: label_dict[labels[int(b["Predicted"])]].convert(b.Before), axis=1)
        import pdb; pdb.set_trace()
        return df

if __name__ == "__main__":

    # Get the name of the file path to store the model
    xgb_model = os.path.join(Constants.prefix, f"models/xgb_sub{round(Constants.limit/1000)}_{Constants.chars}c_{Constants.words}w_{Constants.features}f_v{Constants.version}_model.dat")
    #xgb_model = os.path.join(Constants.prefix, "best/xgb_sub5750_5c_4w_5f_v1_model.dat")
    
    # Preprocess all data and save it under data/custom
    Data.get_data_percentage(1)

    # Train a model using the first x items
    #model = Train.train_limit(Constants.limit)
    # Train a model using some percentage of all data
    #model = Train.train_percentage(percentage)

    # Save the model
    #pickle.dump(model,open(xgb_model,'wb'))
    
    # Load the model
    model = pickle.load(open(xgb_model, "rb"))
    
    # Test either using all data if limit=False, or all data starting from Constants.limit if limit is True
    df = Test.test(model, limit=True)
    import pdb; pdb.set_trace()

    # Convert before tokens using converters and the predicted token class, from the dataframe generated in testing
    Test.convert(df)
    import pdb; pdb.set_trace()
    
    # Plot importance of the model features
    #plot_importance(model, importance_type="gain")
    #pyplot.show()
