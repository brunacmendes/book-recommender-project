# Book Recommender System Project

## Introduction
This project is a partial requirement to the degree of Computer Science by Instituto Federal de Educação, Ciência e Tecnologia de Minas Gerais (IFMG).

## Software 

- python 3.8.2
- tensor flow 2.4.1
- keras 2.4.3
- scikit learn 0.24.1

### Python Virtual Environment (Linux)

To set up a python virtual evironment in linux terminal:

`pip3 install virtualenv`
 
`virtualenv -p python3.8 VirtEnv`

This will place a folder called 'VirtEnv' in the home folder.
To activate it:

`source VirtEnv/bin/activate`

To deactivate the evironment after using it:

`deactivate`

## About the datasets

### Book-crossing (BX)

The Book-Crossing dataset comprises 3 tables.

**BX-Users**

Contains the users. Note that user IDs (`User-ID`) have been anonymized and map to integers. Demographic data is provided (`Location`, `Age`) if available. Otherwise, these fields contain NULL-values.

**BX-Books**

Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (`Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`), i.e., small, medium, large. These URLs point to the Amazon web site.

**BX-Book-Ratings**

Contains the book rating information. Ratings (`Book-Rating`) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

### Amazon Review Data

This Dataset is an updated version of the Amazon review dataset released in 2014. As in the previous version, the complete dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). In addition, this version provides the following features:

- The total number of reviews is 233.1 million (142.8 million in 2014).
- Current data includes reviews in the range May 1996 - Oct 2018.

**Amazon Book Reviews (AB)**

The book dataset used in the project includes only (item, user, rating, timestamp) tuples.
Ratings are expressed on a scale from 0-5 (higher values denoting higher appreciation).


### Where to find
- [Book-Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)


## References

Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books. 

Jianmo Ni, Jiacheng Li, Julian McAuley. Justifying recommendations using distantly-labeled reviews and fined-grained aspects. Empirical Methods in Natural Language Processing (EMNLP), 2019.