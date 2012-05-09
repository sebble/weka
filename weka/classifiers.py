#!/usr/bin/python
"""
2010.2.19 CKS
Light wrapper around Weka.

2011.3.6 CKS
Added method load_raw() to load a raw Weka model file directly.
Added support to retrieving probability distribution of a prediction.
"""
VERSION = (0, 1, 0)
__version__ = '.'.join(map(str, VERSION))

from subprocess import Popen, PIPE
import cPickle as pickle
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import arff

DEFAULT_WEKA_JAR_PATH = '/usr/share/java/weka.jar'#:/usr/share/java/libsvm.jar'

BP = os.path.dirname(os.path.abspath(__file__))
CP = os.environ.get('WEKA_JAR_PATH', DEFAULT_WEKA_JAR_PATH)
for _cp in CP.split(':'):
    assert os.path.isfile(_cp), ("Weka JAR file %s not found. Ensure the " + \
        "file is installed or update your environment's WEKA_JAR_PATH to " + \
        "only include valid locations.") % (_cp,)

# http://weka.sourceforge.net/doc/weka/classifiers/Classifier.html
WEKA_CLASSIFIERS = [
'weka.classifiers.functions.LibSVM -K 0',
'weka.classifiers.functions.LibSVM -K 1',
'weka.classifiers.functions.LibSVM -K 2',
'weka.classifiers.bayes.AODE',
'weka.classifiers.bayes.BayesNet',
'weka.classifiers.bayes.ComplementNaiveBayes',
'weka.classifiers.bayes.NaiveBayes',
'weka.classifiers.bayes.NaiveBayesMultinomial',
'weka.classifiers.bayes.NaiveBayesSimple',
'weka.classifiers.functions.LeastMedSq',
'weka.classifiers.functions.LinearRegression',
'weka.classifiers.functions.Logistic',
'weka.classifiers.functions.MultilayerPerceptron',
'weka.classifiers.functions.PaceRegression',
'weka.classifiers.functions.RBFNetwork',
'weka.classifiers.functions.SMO',
'weka.classifiers.functions.SMOreg',
'weka.classifiers.functions.SimpleLinearRegression',
'weka.classifiers.functions.SimpleLogistic',
'weka.classifiers.functions.VotedPerceptron',
'weka.classifiers.functions.Winnow',
'weka.classifiers.lazy.IB1', 
'weka.classifiers.lazy.IBk',
'weka.classifiers.lazy.KStar',
'weka.classifiers.lazy.LBR',
'weka.classifiers.misc.HyperPipes',
'weka.classifiers.misc.VFI',
'weka.classifiers.rules.ConjunctiveRule',
'weka.classifiers.rules.DecisionTable',
'weka.classifiers.rules.JRip',
'weka.classifiers.rules.NNge',
'weka.classifiers.rules.OneR',
'weka.classifiers.rules.PART',
'weka.classifiers.rules.Prism',
'weka.classifiers.rules.Ridor',
'weka.classifiers.rules.ZeroR',
'weka.classifiers.trees.ADTree',
'weka.classifiers.trees.DecisionStump',
'weka.classifiers.trees.Id3',
'weka.classifiers.trees.J48',
'weka.classifiers.trees.LMT',
'weka.classifiers.trees.NBTree',
'weka.classifiers.trees.REPTree',
'weka.classifiers.trees.RandomForest',
]

class _Helper(object):
    
    def __init__(self, name, ckargs, *args):
        self.name = name
        self.args = [name] + list(args)
        self.ckargs = ckargs
        
    def __call__(self, *args, **kwargs):
        args = list(self.args) + list(args)
        ckargs = self.ckargs
        ckargs.update(kwargs)
        return Classifier(ckargs=ckargs, *args)
    
    def load(self, fn, *args, **kwargs):
        args = list(self.args) + list(args)
        kwargs.update(self.kwargs)
        return Classifier.load(fn, *args, **kwargs)
    
    def __repr__(self):
        return self.name.split('.')[-1]
    
# Generate shortcuts for instantiating each classifier.
for _name in WEKA_CLASSIFIERS:
    _parts = _name.split(' ')
    _name = _parts[0]
    _proper_name = _name.split('.')[-1]
    _ckargs = {}
    _arg_name = None
    for _arg in _parts[1:]:
        if _arg.startswith('-'):
            _arg_name = _arg[1:]
        else:
            _ckargs[_arg_name] = _arg
    _func = _Helper(name=_name, ckargs=_ckargs)
    exec '%s = _func' % _proper_name

# These can be trained incrementally.
# http://weka.sourceforge.net/doc/weka/classifiers/UpdateableClassifier.html
UPDATEABLE_WEKA_CLASSIFIERS = [
'weka.classifiers.bayes.AODE',
'weka.classifiers.lazy.IB1', 
'weka.classifiers.lazy.IBk',
'weka.classifiers.lazy.KStar',
'weka.classifiers.lazy.LWL',
'weka.classifiers.bayes.NaiveBayesUpdateable',
'weka.classifiers.rules.NNge',
'weka.classifiers.meta.RacedIncrementalLogitBoost',
'weka.classifiers.functions.Winnow',
]

WEKA_ACCURACY_REGEX = re.compile('===\s+Stratified cross-validation\s+===' + \
    '\n+\s*\n+\s*Correctly Classified Instances\s+[0-9]+\s+([0-9\.]+)\s+%',
    re.DOTALL)
WEKA_TEST_ACCURACY_REGEX = re.compile('===\s+Error on test data\s+===\n+\s' + \
    '*\n+\s*Correctly Classified Instances\s+[0-9]+\s+([0-9\.]+)\s+%',
    re.DOTALL)

def get_weka_accuracy(arff_fn, arff_test_fn, cls):
    assert cls in WEKA_CLASSIFIERS, "Unknown Weka classifier: %s" % (cls,)
    cmd = "java -cp /usr/share/java/weka.jar:/usr/share/java/libsvm.jar " + \
        "%(cls)s -t \"%(arff_fn)s\" -T \"%(arff_test_fn)s\"" % locals()
    print cmd
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    try:
        acc = float(WEKA_TEST_ACCURACY_REGEX.findall(output)[0])
        return acc
    except IndexError:
        return 0
    except TypeError:
        return 0
    except Exception, e:
        print '!'*80
        print "Unexpected Error: %s" % e
        return 0

class TrainingError(Exception):
    pass

class PredictionError(Exception):
    pass

class Classifier(object):
    
    def __init__(self, name, ckargs=None, model_data=None):
        self._model_data = model_data
        self.name = name # Weka classifier class name.
        self.schema = None
        self.ckargs = ckargs

    @classmethod
    def load(cls, fn, *args, **kwargs):
        if os.path.isfile(fn):
            return pickle.load(open(fn,'rb'))
        else:
            return cls(*args, **kwargs)

    @classmethod
    def load_raw(cls, model_fn, schema, *args, **kwargs):
        """
        Loads a trained classifier from the raw Weka model format.
        Must specify the model schema and classifier name, since
        these aren't currently deduced from the model format.
        """
        c = cls(*args, **kwargs)
        c.schema = schema.copy(schema_only=True)
        c._model_data = open(model_fn,'rb').read()
        return c
        
    def save(self, fn):
        pickle.dump(self, open(fn,'wb'))
        
    def _get_ckargs_str(self):
        ckargs = []
        if self.ckargs:
            for k,v in self.ckargs.iteritems():
                if not k.startswith('-'):
                    k = '-'+k
                ckargs.append('%s %s' % (k,v))
        ckargs = ' '.join(ckargs)
        return ckargs
        
    def train(self, training_data, testing_data=None, verbose=False):
        """
        Updates the classifier with new data.
        """
        model_fn = None
        training_fn = None
        clean_training = False
        testing_fn = None
        clean_testing = False
        try:
            
            # Validate training data.
            if isinstance(training_data, basestring):
                assert os.path.isfile(training_data)
                training_fn = training_data
            else:
                assert isinstance(training_data, arff.ArffFile)
                training_fn = tempfile.mkstemp(suffix='.arff')[1]
                open(training_fn,'w').write(training_data.write())
                clean_training = True
            assert training_fn
                
            # Validate testing data.
            if testing_data:
                if isinstance(testing_data, basestring):
                    assert os.path.isfile(testing_data)
                    testing_fn = testing_data
                else:
                    assert isinstance(testing_data, arff.ArffFile)
                    testing_fn = tempfile.mkstemp(suffix='.arff')[1]
                    open(testing_fn,'w').write(testing_data.write())
                    clean_testing = True
            else:
                testing_fn = training_fn
            assert testing_fn
                
            # Validate model file.
            model_fn = tempfile.mkstemp()[1]
            if self._model_data:
                fout = open(model_fn,'wb')
                fout.write(self._model_data)
                fout.close()
            
            # Call Weka Jar.
            args = dict(CP=CP,
                        classifier_name=self.name,
                        model_fn=model_fn,
                        training_fn=training_fn,
                        testing_fn=testing_fn,
                        ckargs = self._get_ckargs_str(),
                        )
            if self._model_data:
                # Load existing model.
                cmd = "java -cp %(CP)s %(classifier_name)s -l \"%(model_fn)s\" -t \"%(training_fn)s\" -T \"%(testing_fn)s\" -d \"%(model_fn)s\"" % args
            else:
                # Create new model file.
                cmd = "java -cp %(CP)s %(classifier_name)s -t \"%(training_fn)s\" -T \"%(testing_fn)s\" -d \"%(model_fn)s\" %(ckargs)s" % args
            if verbose: print cmd
            p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            if verbose:
                print 'stdout:'
                print stdout_str
                print 'stderr:'
                print stderr_str
            if stderr_str:
                raise TrainingError, stderr_str
            
            # Save schema.
            if not self.schema:
                self.schema = arff.ArffFile.load(training_fn).copy(schema_only=True)
            
            # Save model.
            self._model_data = open(model_fn,'rb').read()
            assert self._model_data
        finally:
            # Cleanup files.
            if model_fn:
                os.remove(model_fn)
            if training_fn and clean_training:
                os.remove(training_fn)
            if testing_fn and clean_testing:
                os.remove(testing_fn)
        
    def predict(self, query_data, verbose=False, distribution=False):
        """
        Iterates over the predicted values and probability (if supported).
        Each iteration yields a tuple of the form (prediction, probability).
        """
        model_fn = None
        query_fn = None
        clean_query = False
        stdout = None
        try:
            
            # Validate query data.
            if isinstance(query_data, basestring):
                assert os.path.isfile(query_data)
                query_fn = query_data
            else:
                assert isinstance(query_data, arff.ArffFile)
                query_fn = tempfile.mkstemp(suffix='.arff')[1]
                open(query_fn,'w').write(query_data.write())
                clean_query = True
            assert query_fn
                
            # Validate model file.
            model_fn = tempfile.mkstemp()[1]
            assert self._model_data, "You must train this classifier before predicting."
            fout = open(model_fn,'wb')
            fout.write(self._model_data)
            fout.close()
                
            # Call Weka Jar.
            args = dict(CP=CP,
                        classifier_name=self.name,
                        model_fn=model_fn,
                        query_fn=query_fn,
                        #ckargs = self._get_ckargs_str(),
                        distribution=('-distribution' if distribution else ''),
                        )
            cmd = "java -cp %(CP)s %(classifier_name)s -p 0 %(distribution)s -l \"%(model_fn)s\" -T \"%(query_fn)s\"" % args
            if verbose: print cmd
            p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            if verbose:
                print 'stdout:'
                print stdout_str
                print 'stderr:'
                print stderr_str
            if stderr_str:
                raise PredictionError, stderr_str
            
            if stdout_str:
                # inst#     actual  predicted error prediction
                #header = 'inst,actual,predicted,error'.split(',')
                query = arff.ArffFile.load(query_fn)
                query_variables = [query.attributes[i] for i,v in enumerate(query.data[0]) if v == arff.MISSING]
                assert query_variables, "There must be at least one query variable in the query."
                header = 'predicted'.split(',')
                # sample line:     1        1:?       4:36   +   1
                
                # Expected output without distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error prediction
                #     1        1:? 11:Acer_tr   +   1

                #=== Predictions on test data ===
                #
                # inst#     actual  predicted      error
                #     1          ?      7              ? 

                # Expected output with distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error distribution
                #     1        1:? 11:Acer_tr   +   0,0,0,0,0,0,0,0,0,0,*1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                
                # Check for distribution output.
                if re.findall('error\s+distribution', stdout_str):
                    matches = re.findall("\s*[0-9\.]+\s+[a-zA-Z0-9\.\?\:]+\s+([a-zA-Z0-9_\.\?\:]+)\s+\+\s+([a-zA-Z0-9\.\?\,\*]+)", stdout_str)
                    assert matches, "No results found matching distribtion pattern in stdout: %s" % stdout_str
                    for match in matches:
                        #print 'match:',match
                        prediction,prob = match
                        class_index,class_label = prediction.split(':')
                        class_index = int(class_index)
                        if distribution:
                            # Convert list of probabilities into a hash linking the prob to the associated class value.
                            prob = dict(zip(query.attribute_data[query.attributes[-1]], map(float, prob.replace('*','').split(','))))
                        else:
                            prob = float(prob)
    #                    if query.attribute_types[query_variables[0]] in ('numeric','real','integer'):
    #                        # Convert match string to data type.
    #                        #class_label = float(class_label)
                        class_label = query.attribute_data[query.attributes[-1]][class_index-1]
                        yield class_label, prob
                else:
                    # Otherwise, assume a simple output.
                    matches = re.findall("\s*[0-9\.]+\s+[a-zA-Z0-9\.\?\:]+\s+([a-zA-Z0-9_\.\?\:]+)\s+", stdout_str)
                    assert matches, "No results found matching simple pattern in stdout: %s" % stdout_str
                    for match in matches:
                        class_index = match
                        class_index = int(class_index)
                        class_name = query.attributes[-1]
                        class_type = query.attribute_types[class_name]
                        if class_type == arff.TYPE_NUMERIC:
                            yield class_index, None
                        else:
                            class_label = query.attribute_data[class_name][class_index-1]
                            yield class_label, None
        finally:
            # Cleanup files.
            if model_fn:
                self._model_data = open(model_fn,'rb').read()
                os.remove(model_fn)
            if query_fn and clean_query:
                os.remove(query_fn)

class Test(unittest.TestCase):
    
    def test_arff(self):
    
        data = arff.ArffFile.load('test/abalone-train.arff')
        self.assertEqual(len(data.attributes), 9)
        
    def test_IBk(self):
        
        # Train a classifier.
        c = Classifier(name='weka.classifiers.lazy.IBk', ckargs={'-K':1})
        training_fn = 'test/abalone-train.arff'
        c.train(training_fn, verbose=0)
        self.assertTrue(c._model_data)
        
        # Make a valid query.
        query_fn = 'test/abalone-query.arff'
        predictions = list(c.predict(query_fn, verbose=0))
        self.assertEqual(predictions[0], (7, None))
            
        # Make a valid query.
        try:
            query_fn = 'test/abalone-query-bad.arff'
            predictions = list(c.predict(query_fn, verbose=0))
            self.assertTrue(0)
        except PredictionError:
            #print 'Invalid query threw exception as expected.'
            self.assertTrue(1)
            
        # Make a valid query manually.
        query = arff.ArffFile(relation='test', schema=[
            ('Sex', ('M','F','I')),
            ('Length', 'real'),
            ('Diameter', 'real'),
            ('Height', 'real'),
            ('Whole weight', 'real'),
            ('Shucked weight', 'real'),
            ('Viscera weight', 'real'),
            ('Shell weight', 'real'),
            ('Class_Rings', 'integer'),
        ])
        query.append(['M',0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,'?'])
        data_str0 = """% 
@relation test
@attribute 'Sex' {M,F,I}
@attribute 'Length' numeric
@attribute 'Diameter' numeric
@attribute 'Height' numeric
@attribute 'Whole weight' numeric
@attribute 'Shucked weight' numeric
@attribute 'Viscera weight' numeric
@attribute 'Shell weight' numeric
@attribute 'Class_Rings' numeric
@data
M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,?
"""
        data_str1 = query.write()
        #print data_str1
        self.assertEqual(data_str0,data_str1)
        predictions = list(c.predict(query, verbose=0))
        self.assertEqual(predictions[0], (7, None))
        
        # Test pickling.
        fn = 'test/IBk.pkl'
        c.save(fn)
        c = Classifier.load(fn)
        predictions = list(c.predict(query, verbose=0))
        self.assertEqual(predictions[0], (7, None))
        #print 'Pickle verified.'
        
        # Make a valid dict query manually.
        query = arff.ArffFile(relation='test',schema=[
            ('Sex', ('M','F','I')),
            ('Length', 'real'),
            ('Diameter', 'real'),
            ('Height', 'real'),
            ('Whole weight', 'real'),
            ('Shucked weight', 'real'),
            ('Viscera weight', 'real'),
            ('Shell weight', 'real'),
            ('Class_Rings', 'integer'),
        ])
        query.append({
            'Sex':'M',
            'Length':0.35,
            'Diameter':0.265,
            'Height':0.09,
            'Whole weight':0.2255,
            'Shucked weight':0.0995,
            'Viscera weight':0.0485,
            'Shell weight':0.07,
            'Class_Rings':arff.MISSING,
        })
        predictions = list(c.predict(query, verbose=0))
        self.assertEqual(predictions[0], (7, None))

    def test_shortcut(self):
        c = IBk(K=1)
        
        training_fn = 'test/abalone-train.arff'
        c.train(training_fn, verbose=0)
        self.assertTrue(c._model_data)
        
        # Make a valid query.
        query_fn = 'test/abalone-query.arff'
        predictions = list(c.predict(query_fn, verbose=0))
        self.assertEqual(predictions[0], (7, None))

if __name__ == '__main__':
    unittest.main()