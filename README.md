# book recommender system project

## Introduction 

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

### Book-crossing

The Book-Crossing dataset comprises 3 tables.

**BX-Users**

Contains the users. Note that user IDs (`User-ID`) have been anonymized and map to integers. Demographic data is provided (`Location`, `Age`) if available. Otherwise, these fields contain NULL-values.

**BX-Books**

Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (`Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`), i.e., small, medium, large. These URLs point to the Amazon web site.

**BX-Book-Ratings**

Contains the book rating information. Ratings (`Book-Rating`) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

### Where to find
- [Book-Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)


## References

Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books. 
