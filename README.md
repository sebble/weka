Weka - Python wrapper for Weka classifiers
=======================================================================

Overview
========

Provides a convenient wrapper for calling Weka classifiers from Python.

Installation
------------

First install the Weka and LibSVM Java libraries. On Debian/Ubuntu this is simply:

    sudo apt-get install weka libsvm-java

Then install the Python package with pip:

    sudo pip install weka

Usage
-----

Train and test a Weka classifier by instantiating the Classifier class,
passing in the name of the classifier you want to use:

    from weka.classifiers import Classifier
    c = Classifier(name='weka.classifiers.lazy.IBk', ckargs={'-K':1})
    c.train('training.arff')
    predictions = c.predict('query.arff')

Alternatively, you can instantiate the classifier by calling its name directly:

    from weka.classifiers import IBk
    c = IBk(K=1)
    c.train('training.arff')
    predictions = c.predict('query.arff')

The instance contains Weka's serialized model, so the classifier can be easily
pickled and unpickled like any normal Python instance:

    c.save('myclassifier.pkl')
    c = Classifier.load('myclassifier.pkl')
    predictions = c.predict('query.arff')
    