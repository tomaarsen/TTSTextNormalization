# TTSTextNormalization

This repository houses my solution to Google's [Text Normalization Challenge - English Language](https://www.kaggle.com/c/text-normalization-challenge-english-language). Most of the magic happens within the converter directory, which is responsible for the actual conversions from input to output tokens.
Alongside the code is a [paper](https://github.com/tomaarsen/TTSTextNormalization/blob/master/paper.pdf) written regarding my solution. The abstract for this paper is as follows:

---

## Abstract
This paper proposes a method for solving, as well as a solution to, a text-to-speech normalization problem, which focuses on converting text from written expressions into spoken forms. The method parses input tokens through a gradient boosted decision tree model, which classifies the token as one of 16 different types of tokens. The token is then converted based on the predicted token type, resulting in a normalized output of the spoken form. Upon entering a related text-to-speech normalization competition, the solution achieved an accuracy of **99.590%**, placing 12th out of the 260 teams, or within the **top 5%** of all submissions.

---

In order to run any of the python files, the `data/raw` folder must contain the raw training and testing data from the competition itself. Due to the Terms and Conditions of the competition, this data cannot be shared on this repository.

This repository acts as an archive, and is not intended to be updated.

---

### Contributing
I am not taking contributions for this repository, as it is designed as an archive.

---

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
