# Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
2011.3.6 CKS
Fixed regular expression to handle special characters in attribute names.
Added options for parsing and copying only the schema.
2012.11.4 CKS
Added support for streaming and sparse data.
"""

import os
import sys
import re
import copy
import unittest
import tempfile
from decimal import Decimal
from io import StringIO

MISSING = '?'

def is_numeric(v):
    try:
        float(v)
        return True
    except:
        return False

DENSE = 'dense'
SPARSE = 'sparse'
FORMATS = (DENSE, SPARSE)

TYPE_INTEGER = 'integer'
TYPE_NUMERIC = 'numeric' # float or integer
TYPE_REAL = 'real'
TYPE_STRING = 'string'
TYPE_NOMINAL = 'nominal'
TYPES = (
    TYPE_INTEGER,
    TYPE_NUMERIC,
    TYPE_STRING,
    TYPE_NOMINAL,
)
NUMERIC_TYPES = (
    TYPE_INTEGER,
    TYPE_NUMERIC,
    TYPE_REAL,
)

STRIP_QUOTES_REGEX = re.compile('^[\'\"]|[\'\"]$')

class Value(object):
    """
    Base helper class for tagging units of data with an explicit schema type.
    """
    
    __slots__ = ('value', 'cls')
    
    def __init__(self, v, cls=False):
        self.value = v
        self.cls = cls
    
    def __hash__(self):
        #return hash((self.value, self.cls))
        return hash(self.value)
    
    def __cmp__(self, other):
        if type(self) == type(other):
            #return cmp((self.value, self.cls), (other.value, other.cls))
            return cmp(self.value, other.value)
        else:
            return NotImplemented
    
    def __repr__(self):
        return repr(self.value)

class Integer(Value):
    c_type = TYPE_INTEGER
    def __init__(self, v, *args, **kwargs):
        v = int(v)
        super(Integer, self).__init__(v, *args, **kwargs)
Int = Integer

class Numeric(Value):
    c_type = TYPE_NUMERIC
    def __init__(self, v, *args, **kwargs):
        # TODO:causes loss of precision?
        v = float(v)
        super(Numeric, self).__init__(v, *args, **kwargs)
Num = Numeric

class String(Value):
    c_type = TYPE_STRING
    def __init__(self, v, *args, **kwargs):
        v = str(v)
        super(String, self).__init__(v, *args, **kwargs)
Str = String

class Nominal(Value):
    c_type = TYPE_NOMINAL
    pass
Nom = Nominal

TYPE_TO_CLASS = {
    TYPE_INTEGER: Integer,
    TYPE_NUMERIC: Numeric,
    TYPE_STRING: String,
    TYPE_NOMINAL: Nominal,
    TYPE_REAL: Numeric,
}

def wrap_value(v):
    if isinstance(v, Value):
        return v
    if v == MISSING:
        return Str(v)
    if isinstance(v, str):
        return Str(v)
    try:
        return Num(v)
    except ValueError:
        pass
    try:
        return Int(v)
    except ValueError:
        pass

class ArffFile(object):
    """An ARFF File object describes a data set consisting of a number
    of data points made up of attributes. The whole data set is called
    a 'relation'. Supported attributes are:

    - 'numeric': floating point numbers
    - 'string': strings
    - 'nominal': taking one of a number of possible values

    Not all features of ARFF files are supported yet. The most notable
    exceptions are:

    - no sparse data
    - no support for date and relational attributes

    Also, parsing of strings might still be a bit brittle.

    You can either load or save from files, or write and parse from a
    string.

    You can also construct an empty ARFF file and then fill in your
    data by hand. To define attributes use the define_attribute method.

    Attributes are:

    - 'relation': name of the relation
    - 'attributes': names of the attributes
    - 'attribute_types': types of the attributes
    - 'attribute_data': additional data, for example for nominal attributes.
    - 'comment': the initial comment in the file. Typically contains some
                 information on the data set.
    - 'data': the actual data, by data points.
    """
    def __init__(self, relation='', schema=None):
        """Construct an empty ARFF structure."""
        self.relation = relation
        self.clear()
        
        # Load schema.
        if schema:
            for name,data in schema:
                name = STRIP_QUOTES_REGEX.sub('', name)
                self.attributes.append(name)
                if type(data) in (tuple,list):
                    self.attribute_types[name] = TYPE_NOMINAL
                    self.attribute_data[name] = set(data)
                else:
                    self.attribute_types[name] = data
                    self.attribute_data[name] = None
        
    def clear(self):
        self.attributes = [] # [attr_name, attr_name, ...]
        self.attribute_types = dict() # {attr_name:type}
        self.attribute_data = dict() # {attr_name:[nominal values]}
        self._filename = None
        self.comment = []
        self.data = []
        self.lineno = 0
        self.fout = None
        self.class_attr_name = None
    
    def get_attribute_value(self, name, index):
        """
        Returns the value associated with the given value index
        of the attribute with the given name.
        
        This is only applicable for nominal and string types.
        """
        if index == MISSING:
            return
        elif self.attribute_types[name] in NUMERIC_TYPES:
            at = self.attribute_types[name]
            if at == TYPE_INTEGER:
                return int(index)
            else:
                return Decimal(str(index))
        else:
            assert self.attribute_types[name] == TYPE_NOMINAL
            cls_index, cls_value = index.split(':')
            #return self.attribute_data[name][index-1]
            if cls_value != MISSING:
                assert cls_value in self.attribute_data[name], \
                    'Predicted value "%s" but only values %s are allowed.' \
                        % (cls_value, ', '.join(self.attribute_data[name]))
            return cls_value
    
    def __iter__(self):
        for d in self.data:
            named = dict(list(zip(
                [re.sub('^[\'\"]|[\'\"]$', '', _) for _ in self.attributes],
                d)))
            assert len(d) == len(self.attributes)
            assert len(d) == len(named)
            yield named

    @classmethod
    def load(cls, filename, schema_only=False):
        """
        Load an ARFF File from a file.
        """
        o = open(filename)
        s = o.read()
        a = cls.parse(s, schema_only=schema_only)
        if not schema_only:
            a._filename = filename
        o.close()
        return a

    @classmethod
    def parse(cls, s, schema_only=False):
        """
        Parse an ARFF File already loaded into a string.
        """
        a = cls()
        a.state = 'comment'
        a.lineno = 1
        for l in s.splitlines():
            a.parseline(l)
            a.lineno += 1
            if schema_only and a.state == 'data':
                # Don't parse data if we're only loading the schema.
                break
        return a

    def copy(self, schema_only=False):
        """
        Creates a deepcopy of the instance.
        If schema_only is True, the data will be excluded from the copy.
        """
        o = type(self)()
        o.relation = self.relation
        o.attributes = list(self.attributes)
        o.attribute_types = self.attribute_types.copy()
        o.attribute_data = self.attribute_data.copy()
        if not schema_only:
            o.comment = list(self.comment)
            o.data = copy.deepcopy(self.data)
        return o

    def flush(self):
        if self.fout:
            self.fout.flush()

    def open_stream(self, class_attr_name=None, fn=None):
        """
        Save an arff structure to a file, leaving the file object
        open for writing of new data samples.
        This prevents you from directly accessing the data via Python,
        but when generating a huge file, this prevents all your data
        from being stored in memory.
        """
        if fn:
            self.fout_fn = fn
        else:
            fd, self.fout_fn = tempfile.mkstemp()
            os.close(fd)
        self.fout = open(self.fout_fn, 'w')
        if class_attr_name:
            self.class_attr_name = class_attr_name
        self.write(fout=self.fout, schema_only=True)
        self.write(fout=self.fout, data_only=True)
        self.fout.flush()
        
    def close_stream(self):
        """
        Terminates an open stream and returns the filename
        of the file containing the streamed data.
        """
        if self.fout:
            fout = self.fout
            fout_fn = self.fout_fn
            self.fout.flush()
            self.fout.close()
            self.fout = None
            self.fout_fn = None
            return fout_fn

    def save(self, filename=None):
        """
        Save an arff structure to a file.
        """
        filename = filename or self._filename
        o = open(filename, 'w')
        o.write(self.write())
        o.close()

    def write_line(self, d, format=SPARSE):
        """
        Converts a single data line to a string.
        """
        if format == DENSE:
            #TODO:fix
            assert not isinstance(d, dict), NotImplemented
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at in NUMERIC_TYPES:
                    line.append(str(e))
                elif at == TYPE_STRING:
                    line.append(self.esc(e))
                elif at == TYPE_NOMINAL:
                    line.append(e)
                else:
                    raise Exception("Type " + at + " not supported for writing!")
            s = ','.join(map(str, line))
            return s
        elif format == SPARSE:
            line = []
            
            # Convert flat row into dictionary.
            if isinstance(d, (list, tuple)):
                d = dict(list(zip(self.attributes, d)))
                for k in d.keys():
                    at = self.attribute_types.get(k)
                    if isinstance(d[k], Value):
                        continue
                    elif d[k] == MISSING:
                        d[k] = Str(d[k])
                    elif at in (TYPE_NUMERIC, TYPE_REAL):
                        d[k] = Num(d[k])
                    elif at == TYPE_STRING:
                        d[k] = Str(d[k])
                    elif at == TYPE_INTEGER:
                        d[k] = Int(d[k])
                    elif at == TYPE_NOMINAL:
                        d[k] = Nom(d[k])
                    else:
                        raise Exception('Unknown type: %s' % at)
                    
            for i, name in enumerate(self.attributes):
                v = d.get(name)
                if v is None:
#                    print 'Skipping attribute with None value:', name
                    continue
                elif v == MISSING or (isinstance(v, Value) \
                and v.value == MISSING):
                    v = MISSING
                elif isinstance(v, String):
                    v = '"%s"' % v.value
                elif isinstance(v, Value):
                    v = v.value
                    
                if v != MISSING and self.attribute_types[name] == TYPE_NOMINAL\
                and str(v) not in list(map(str, self.attribute_data[name])):
#                    print 'Skipping nominal attribute with unregistered value.'
#                    print 'Nominal attribute %s is set to "%s" but only allows %s.' % (name, v, ', '.join(map(str, self.attribute_data[name])))
#                    print [type(_) for _ in ([v]+list(self.attribute_data[name]))]
                    pass
                else:
                    line.append('%i %s' % (i, v))
            if len(line) == 1 and MISSING in line[-1]:
                # Skip lines with nothing other than a missing class.
                return
            elif not len(line):
                # Don't write blank lines.
                return
            return '{' + (', '.join(line)) + '}'
        else:
            raise Exception('Uknown format: %s' % (format,))

    def write_attributes(self, fout=None):
        close = False
        if fout is None:
            close = True
            fout = StringIO()
        for a in self.attributes:
            at = self.attribute_types[a]
            if at == TYPE_INTEGER:
                print("@attribute " + self.esc(a) + " integer", file=fout)
            elif at in (TYPE_NUMERIC, TYPE_REAL):
                print("@attribute " + self.esc(a) + " numeric", file=fout)
            elif at == TYPE_STRING:
                print("@attribute " + self.esc(a) + " string", file=fout)
            elif at == TYPE_NOMINAL:
                nom_vals = [_ for _ in self.attribute_data[a] if _ != MISSING]
                nom_vals = sorted(nom_vals)
                print("@attribute " + self.esc(a) + \
                    " {" + ','.join(map(str, nom_vals)) + "}", file=fout)
            else:
                raise Exception("Type " + at + " not supported for writing!")
        if isinstance(fout, StringIO) and close:
            return fout.getvalue()

    def write(self,
        fout=None,
        format=SPARSE,
        schema_only=False,
        data_only=False):
        """
        Write an arff structure to a string.
        """
        assert not (schema_only and data_only), 'Make up your mind.'
        assert format in FORMATS, \
            'Invalid format "%s". Should be one of: %s' \
                % (format, ', '.join(FORMATS))
        close = False
        if fout is None:
            close = True
            fout = StringIO()
        if not data_only:
            print('% ' + re.sub("\n", "\n% ", '\n'.join(self.comment)), file=fout)
            print("@relation " + self.relation, file=fout)
            self.write_attributes(fout=fout)
        if not schema_only:
            print("@data", file=fout)
            for d in self.data:
                line_str = self.write_line(d, format=format)
                if line_str:
                    print(line_str, file=fout)
        if isinstance(fout, StringIO) and close:
            return fout.getvalue()

    def esc(self, s):
        """
        Escape a string if it contains spaces.
        """
        return ("\'" + s + "\'").replace("''","'")

    def define_attribute(self, name, atype, data=None):
        """
        Define a new attribute. atype has to be one
        of 'integer', 'real', 'numeric', 'string', and 'nominal'.
        For nominal attributes, pass the possible values as data.
        """
        self.attributes.append(name)
        assert atype in TYPES, \
            "Unknown type '%s'. Must be one of: %s" % \
            (atype, ', '.join(TYPES),)
        self.attribute_types[name] = atype
        self.attribute_data[name] = data

    def parseline(self, l):
        if self.state == 'comment':
            if len(l) > 0 and l[0] == '%':
                self.comment.append(l[2:])
            else:
                self.comment = '\n'.join(self.comment)
                self.state = 'in_header'
                self.parseline(l)
        elif self.state == 'in_header':
            ll = l.lower()
            if ll.startswith('@relation '):
                self.__parse_relation(l)
            if ll.startswith('@attribute '):
                self.__parse_attribute(l)
            if ll.startswith('@data'):
                self.state = 'data'
        elif self.state == 'data':
            if len(l) > 0 and l[0] != '%':
                self._parse_data(l)

    def __parse_relation(self, l):
        l = l.split()
        self.relation = l[1]

    def __parse_attribute(self, l):
        p = re.compile(
            r'[a-zA-Z_][a-zA-Z0-9_\-\[\]]*|\{[^\}]*\}|\'[^\']+\'|\"[^\"]+\"')
        l = [s.strip() for s in p.findall(l)]
        name = l[1]
        name = STRIP_QUOTES_REGEX.sub('', name)
        atype = l[2]#.lower()
        if atype == TYPE_INTEGER:
            self.define_attribute(name, TYPE_INTEGER)
        elif (atype == TYPE_REAL or atype == TYPE_NUMERIC):
            self.define_attribute(name, TYPE_NUMERIC)
        elif atype == TYPE_STRING:
            self.define_attribute(name, TYPE_STRING)
        elif atype[0] == '{' and atype[-1] == '}':
            values = [s.strip () for s in atype[1:-1].split(',')]
            self.define_attribute(name, TYPE_NOMINAL, values)
        else:
            print("Unsupported type " + atype + " for attribute " + name + ".")

    def _parse_data(self, l):
        if isinstance(l, str):
            l = l.strip()
            if l.startswith('{'):
                assert l.endswith('}'), 'Malformed sparse data line: %s' % (l,)
                assert not self.fout, NotImplemented
                dline = {}
                parts = re.split(r'(?<!\\),', l[1:-1])
                for part in parts:
                    index, value = re.findall(
                        '(^[0-9]+)\s+(.*)$', part.strip())[0]
                    index = int(index)
                    if value[0] == value[-1] and value[0] in ('"', "'"):
                        # Strip quotes.
                        value = value[1:-1]
                    # TODO:0 or 1-indexed? Weka uses 0-indexing?!
                    #name = self.attributes[index-1]
                    name = self.attributes[index]
                    ValueClass = TYPE_TO_CLASS[self.attribute_types[name]]
                    if value == MISSING:
                        dline[name] = Str(value)
                    else:
                        dline[name] = ValueClass(value)
                self.data.append(dline)
                return
            else:
                # Convert string to list.
                l = [s.strip() for s in l.split(',')]
        elif isinstance(l, dict):
            assert len(l) == len(self.attributes), \
                "Sparse data not supported."
            # Convert dict to list.
            #l = dict((k,v) for k,v in l.iteritems())
            # Confirm complete feature name overlap.
            assert set(self.esc(a) for a in l) == \
                set(self.esc(a) for a in self.attributes)
            l = [l[name] for name in self.attributes]
        else:
            # Otherwise, confirm list.
            assert type(l) in (tuple,list)
        if len(l) != len(self.attributes):
            print(("Warning: line %d contains %i " + \
                   "values but it should contain %i values") \
                % (self.lineno, len(l), len(self.attributes)))
            return 

        datum = []
        for n, v in zip(self.attributes, l):
            at = self.attribute_types[n]
            if v == MISSING:
                datum.append(v)
            elif at == TYPE_INTEGER:
                datum.append(int(v))
            elif at in (TYPE_NUMERIC, TYPE_REAL):
                datum.append(Decimal(str(v)))
            elif at == TYPE_STRING:
                datum.append(v)
            elif at == TYPE_NOMINAL:
                if v in self.attribute_data[n]:
                    datum.append(v)
                else:
                    raise Exception('Incorrect value %s for nominal attribute %s' % (v, n))
        if self.fout:
            # If we're streaming out data, then don't even bother saving it to
            # memory and just flush it out to disk instead.
            line_str = self.write_line(datum)
            if line_str:
                print(line_str, file=self.fout)
            self.fout.flush()
        else:
            self.data.append(datum)

    def __print_warning(self, msg):
        print(('Warning (line %d): ' % self.lineno) + msg)

    def dump(self):
        """Print an overview of the ARFF file."""
        print("Relation " + self.relation)
        print("  With attributes")
        for n in self.attributes:
            if self.attribute_types[n] != TYPE_NOMINAL:
                print("    %s of type %s" % (n, self.attribute_types[n]))
            else:
                print(("    " + n + " of type nominal with values " +
                       ', '.join(self.attribute_data[n])))
        for d in self.data:
            print(d)
    
    def set_class(self, name):
        assert name in self.attributes
        self.attributes.remove(name)
        self.attributes.append(name)
    
    def set_nominal_values(self, name, values):
        assert name in self.attributes
        assert self.attribute_types[name] == TYPE_NOMINAL
        self.attribute_data.setdefault(name, set())
        self.attribute_data[name] = set(self.attribute_data[name])
        self.attribute_data[name].update(values)
    
    def append(self, line, schema_only=False, update_schema=True):
        schema_change = False
        if isinstance(line, dict):
            # Validate line types against schema.
            if update_schema:
                for k, v in list(line.items()):
                    prior_type = self.attribute_types.get(k,
                        v.c_type if isinstance(v, Value) else None)
                    if not isinstance(v, Value):
                        if v == MISSING:
                            v = Str(v)
                        else:
                            v = TYPE_TO_CLASS[prior_type](v)
                    if v.value != MISSING:
                        assert prior_type == v.c_type, \
                            ('Attempting to set attribute %s to type %s but ' + \
                             'it is already defined as type %s.') \
                                % (k, prior_type, v.c_type)
                    if k not in self.attribute_types:
                        if self.fout:
                            # Remove feature that violates the schema
                            # during streaming.
                            if k in line: del line[k]
                        else:
                            self.attribute_types[k] = v.c_type
                            self.attributes.append(k)
                            schema_change = True
                    if isinstance(v, Nominal):
                        if self.fout:
                            # Remove feature that violates the schema
                            # during streaming.
                            if k not in self.attributes:
                                if k in line: del line[k]
                            elif v.value not in self.attribute_data[k]:
                                if k in line: del line[k]
                        else:
                            self.attribute_data.setdefault(k, set())
                            if v.value not in self.attribute_data[k]:
                                self.attribute_data[k].add(v.value)
                                schema_change = True
                    if v.cls:
                        if self.class_attr_name is None:
                            self.class_attr_name = k
                        else:
                            assert self.class_attr_name == k, \
                                ('Attempting to set class to "%s" when it has ' + \
                                'already been set to "%s"') \
                                    % (k, self.class_attr_name)
                                    
                    # Ensure the class attribute is the last one listed,
                    # as that's assumed to be the class unless otherwise specified.
                    if self.class_attr_name:
                        try:
                            self.attributes.remove(self.class_attr_name)
                            self.attributes.append(self.class_attr_name)
                        except ValueError:
                            pass
                    
                if schema_change:
                    assert not self.fout, \
                        'Attempting to add data that doesn\'t match ' + \
                        'the schema while streaming.'
                    
            if not schema_only:
                # Append line to data set.
                if self.fout:
#                    print 'line:',line
#                    for k in line.iterkeys():
#                        print k, k in self.attribute_types
#                    print '-'*80
                    line_str = self.write_line(line)
                    if line_str:
                        print(line_str, file=self.fout)
                else:
                    self.data.append(line)
        else:
            assert len(line) == len(self.attributes)
            self._parse_data(line)

class Test(unittest.TestCase):
    
    def test_sparse_stream(self):
        
        s0 = """% 
@relation test-abalone
@attribute 'Diameter' numeric
@attribute 'Length' numeric
@attribute 'Sex' {F,M}
@attribute 'Whole_weight' numeric
@attribute 'Class_Rings' integer
@data
{0 0.286, 1 0.35, 2 M, 4 15}
{0 0.86, 2 F, 3 0.98, 4 7}
"""
        
        rows = [
            dict(
                Sex=Nom('M'),
                Length=Num(0.35),
                Diameter=Num(0.286),
                Class_Rings=Int(15, cls=True)),
            dict(
                Sex=Nom('F'),
                Diameter=Num(0.86),
                Whole_weight=Num(0.98),
                Class_Rings=Int(7, cls=True)),
        ]
        rows_extra = [
            dict(
                Sex=Nom('N'),
                Length=Num(0.35),
                Diameter=Num(0.286),
                Class_Rings=Int(15, cls=True)),
            dict(
                Sex=Nom('B'),
                Diameter=Num(0.86),
                Whole_weight=Num(0.98),
                Class_Rings=Int(7, cls=True)),
        ]
        
        a1 = ArffFile(relation='test-abalone')
        for row in rows:
            a1.append(row)
        self.assertEqual(a1.class_attr_name, 'Class_Rings')
        s1 = a1.write()
#        print s0
#        print s1
        self.assertEqual(s1, s0)
        
        a2 = ArffFile.parse(s1)
        for i,line in enumerate(a2.data):
            self.assertEqual(line, a1.data[i])
        s2 = a2.write()
        self.assertEqual(s1, s2)
        
        a3 = ArffFile(relation='test-abalone')
        self.assertEqual(len(a3.data), 0)
        #a3.open_stream(class_attr_name='Class_Rings')
        
        # When streaming, you have to provide your schema ahead of time,
        # since otherwise we'd have to update the indexes on all files
        # previously written to the file.
        for row in rows:
            a3.append(row, schema_only=True)
            self.assertEqual(len(a3.data), 0)
            
        a3.open_stream(class_attr_name='Class_Rings')
        for row in (rows+rows_extra):
            a3.append(row)
            self.assertEqual(len(a3.data), 0)
        
        fn = a3.close_stream()
        s3 = open(fn,'r').read()
        #print s3
        os.remove(fn)
        # Note the rows that have features violating the schema are
        # automatically omitted when in streaming mode.
        self.assertEqual(s3, """% 
@relation test-abalone
@attribute 'Diameter' numeric
@attribute 'Length' numeric
@attribute 'Sex' {F,M}
@attribute 'Whole_weight' numeric
@attribute 'Class_Rings' integer
@data
{0 0.286, 1 0.35, 2 M, 4 15}
{0 0.86, 2 F, 3 0.98, 4 7}
{0 0.286, 1 0.35, 4 15}
{0 0.86, 3 0.98, 4 7}
""")

if __name__ == '__main__':
    unittest.main()
