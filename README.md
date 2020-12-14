# native-language-recognition

## Introduction
The goal of this project is to build an automatic speech recognition system that can recognize the native language of the speaker from English utterances. 
The cultural differences between non-native English speakers might allow virtual assistants to further understand the context between a speakers question or request.
This project is greatly motivated by a recent research challenge, which made a large data-set available to test and develop such a method. 
From an initial goal of reproducing the baseline results, our goal is to gain a deeper understanding into the techniques required to recognize these slight differences.

## Data set

The dataset used is the Educational Testing Service Corpus of Non-Native Spoken English which is made of English utterances of 45 seconds from eleven different backgrounds.
There are 3300 training instances (41.3 h, ∼ 64%), 965 validation instances (12.1 h,
∼ 19%) and 867 testing instances (10.8 h, ∼ 17%). In total, there are 5132 unique nonnative speakers. 
The following table demonstrates the distribution of all samples.
<div id="tab:interspeech_dataset_description">

| **L1**     | **Training** | **Development** | **Test** | **Sum** |
|:-----------|:------------:|:---------------:|:--------:|:-------:|
| Arabic     |     300      |       86        |    80    |   466   |
| Chinese    |     300      |       84        |    74    |   458   |
| French     |     300      |       80        |    78    |   458   |
| German     |     300      |       85        |    75    |   460   |
| Hindi      |     300      |       83        |    82    |   465   |
| Italian    |     300      |       94        |    68    |   462   |
| Japanese   |     300      |       85        |    75    |   460   |
| Korean     |     300      |       90        |    80    |   470   |
| Spanish    |     300      |       100       |    77    |   477   |
| Telugu     |     300      |       83        |    88    |   471   |
| Turkish    |     300      |       95        |    90    |   485   |
| **TOTAL:** |     3300     |       965       |   867    |  5132   |

Number of 45 second recordings for each of the L1 languages.

</div>

## Methods
Our goals during this project includes the following approaches:
* Replicate the [SVM based baseline](https://github.com/albertonietos/native-language-recognition/blob/main/baseline.py) from the INTERSPEECH paper.
* Simplify the ComParE dataset using [feature selection and dimensionality reduction](https://github.com/albertonietos/native-language-recognition/blob/main/feature_selection.py) algorithms:
  - Variance thresholding, Chi-squared, Relief-F, [Principal component analysis](https://github.com/albertonietos/native-language-recognition/blob/main/feature-based-methods/PCA.py)
* Trying other (possibly better) [classifiers](https://github.com/albertonietos/native-language-recognition/tree/main/feature-based-methods): boosting, tree based methods and DNNs.
* Evaluate feasibility of an [end-to-end deep learning approach](https://github.com/albertonietos/native-language-recognition/tree/main/end2you) using raw audio.
* Evaluate the feasibility of [image-based systems](https://github.com/albertonietos/native-language-recognition/tree/main/image-based-methods) (spectograms).
