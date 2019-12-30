
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

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

class TestConverter:
    def __init__(self):
        super().__init__()

        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/raw/en_train.csv"))

        labels = {
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

        total = df.shape[0]
        total_incorrect = 0
        for label in labels:
            converter = labels[label]

            cls_df = df[df["class"] == label]
            cls_df["CustAfter"] = cls_df.before.apply(converter.convert)

            incorrect = cls_df[cls_df["after"] != cls_df["CustAfter"]].shape[0]
            total_incorrect += incorrect

            print(f"Class: {label}")
            print(f"Total: {cls_df.shape[0]}")
            print(f"Wrong: {incorrect}")
            print(f"Accuracy: {100 - incorrect / cls_df.shape[0] * 100:.4f}%\n")

        print(f"Total wrong: {total_incorrect}")
        print(f"Total entries: {total}")
        print(f"Total Accuracy: {100 - total_incorrect / total * 100:.6f}%\n")
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    TestConverter()
