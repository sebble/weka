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
"""

import sys
import re
import copy
from decimal import Decimal

MISSING = '?'

def is_numeric(v):
    try:
        float(v)
        return True
    except:
        return False

TYPE_NUMERIC = 'numeric'
TYPE_STRING = 'string'
TYPE_NOMINAL = 'nominal'
TYPES = [TYPE_NUMERIC,TYPE_STRING,TYPE_NOMINAL]

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
                name = re.sub('^[\'\"]|[\'\"]$', '', name)
                self.attributes.append(name)
                if type(data) in (tuple,list):
                    self.attribute_types[name] = TYPE_NOMINAL
                    self.attribute_data[name] = list(data)
                else:
                    self.attribute_types[name] = data
                    self.attribute_data[name] = None
        
    def clear(self):
        self.attributes = []
        self.attribute_types = dict()
        self.attribute_data = dict()
        self._filename = None
        self.comment = []
        self.data = []
        self.lineno = 0
        self.fout = None
        
    def __iter__(self):
        for d in self.data:
#            print '__iter0:',len(d),len(self.attributes)
#            from collections import defaultdict
#            counts = defaultdict(int)
#            for n in self.attributes:
#                counts[n] += 1
#            print 'dups:',[(k,v) for k,v in counts.items() if v > 1]
#            sys.exit()
            named = dict(zip(self.attributes, d))
            assert len(d) == len(self.attributes)
            assert len(d) == len(named)
            yield named

    @classmethod
    def load(cls, filename, schema_only=False):
        """Load an ARFF File from a file."""
        o = open(filename)
        s = o.read()
        a = cls.parse(s, schema_only=schema_only)
        if not schema_only:
            a._filename = filename
        o.close()
        return a

    @classmethod
    def parse(cls, s, schema_only=False):
        """Parse an ARFF File already loaded into a string."""
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

    def save_stream(self, filename):
        """Save an arff structure to a file, leaving the file object
        open for writing of new data samples."""
        if isinstance(filename, basestring):
            filename = filename or self._filename
            self.fout = open(filename, 'w')
        else:
            self.fout = filename
        print>>self.fout, self.write().strip()
        self.fout.flush()
        
    def close_stream(self):
        if self.fout:
            self.fout.close()
            self.fout = None

    def save(self, filename=None):
        """Save an arff structure to a file."""
        filename = filename or self._filename
        o = open(filename, 'w')
        o.write(self.write())
        o.close()

    def write_line(self, d):
        """
        Converts a single data line to a string.
        """
        line = []
        for e, a in zip(d, self.attributes):
            at = self.attribute_types[a]
            if at in (TYPE_NUMERIC,'real','integer'):
                line.append(str(e))
            elif at == TYPE_STRING:
                line.append(self.esc(e))
            elif at == TYPE_NOMINAL:
                line.append(e)
            else:
                raise Exception, "Type " + at + " not supported for writing!"
        s = ','.join(map(str,line))
        return s

    def write(self):
        """Write an arff structure to a string."""
        o = []
        o.append('% ' + re.sub("\n", "\n% ", '\n'.join(self.comment)))
        o.append("@relation " + self.relation)
        for a in self.attributes:
            at = self.attribute_types[a]
            if at in (TYPE_NUMERIC,'real','integer'):
                o.append("@attribute " + self.esc(a) + " numeric")
            elif at == TYPE_STRING:
                o.append("@attribute " + self.esc(a) + " string")
            elif at == TYPE_NOMINAL:
                #print a,self.attribute_data[a]
                o.append("@attribute " + self.esc(a) +
                         " {" + ','.join(map(str, self.attribute_data[a])) + "}")
            else:
                raise Exception, "Type " + at + " not supported for writing!"
        o.append("@data")
        for d in self.data:
            o.append(self.write_line(d))
        return "\n".join(o) + "\n"

    def esc(self, s):
        "Escape a string if it contains spaces"
        return ("\'" + s + "\'").replace("''","'")

    def define_attribute(self, name, atype, data=None):
        """Define a new attribute. atype has to be one
        of 'numeric', 'string', and 'nominal'. For nominal
        attributes, pass the possible values as data."""
        self.attributes.append(name)
        assert atype in TYPES, "Unknown type '%s'. Must be one of: %s" % (','.join(TYPES),)
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
        p = re.compile(r'[a-zA-Z_][a-zA-Z0-9_\[\]]*|\{[^\}]*\}|\'[^\']+\'|\"[^\"]+\"')
        l = [s.strip() for s in p.findall(l)]
        name = l[1]
        atype = l[2]#.lower()
        if (atype == 'real' or
            atype == TYPE_NUMERIC or
            atype == 'integer'):
            self.define_attribute(name, TYPE_NUMERIC)
        elif atype == TYPE_STRING:
            self.define_attribute(name, TYPE_STRING)
        elif atype[0] == '{' and atype[-1] == '}':
            values = [s.strip () for s in atype[1:-1].split(',')]
            self.define_attribute(name, TYPE_NOMINAL, values)
        else:
            print "Unsupported type " + atype + " for attribute " + name + "."

    def _parse_data(self, l):
        if isinstance(l, basestring):
            # Convert string to list.
            l = [s.strip() for s in l.split(',')]
        elif isinstance(l, dict):
            assert len(l) == len(self.attributes), "Sparse data not supported."
            # Convert dict to list.
            #l = dict((k,v) for k,v in l.iteritems())
            # Confirm complete feature name overlap.
            assert set(self.esc(a) for a in l) == set(self.esc(a) for a in self.attributes)
            l = [l[name] for name in self.attributes]
        else:
            # Otherwise, confirm list.
            assert type(l) in (tuple,list)
        if len(l) != len(self.attributes):
            print "Warning: line %d contains wrong number of values" % self.lineno
            return 

        datum = []
        for n, v in zip(self.attributes, l):
            at = self.attribute_types[n]
            if v == MISSING:
                datum.append(v)
            elif at in (TYPE_NUMERIC,'real','integer'):
                if is_numeric(v) or re.match(r'[+-]?[0-9]+(?:\.[0-9]*(?:[eE]-?[0-9]+)?)?', v):
                    if at == 'integer':
                        datum.append(int(v))
                    else:
                        datum.append(Decimal(str(v)))
                else:
                    raise Exception, 'non-numeric value %s for numeric attribute %s' % (v, n)
            elif at == TYPE_STRING:
                datum.append(v)
            elif at == TYPE_NOMINAL:
                if v in self.attribute_data[n]:
                    datum.append(v)
                else:
                    raise Exception, 'incorrect value %s for nominal attribute %s' % (v, n)
        if self.fout:
            # If we're streaming out data, then don't even bother saving it to
            # memory and just flush it out to disk instead.
            print>>self.fout, self.write_line(datum)
            self.fout.flush()
        else:
            self.data.append(datum)

    def __print_warning(self, msg):
        print ('Warning (line %d): ' % self.lineno) + msg

    def dump(self):
        """Print an overview of the ARFF file."""
        print "Relation " + self.relation
        print "  With attributes"
        for n in self.attributes:
            if self.attribute_types[n] != TYPE_NOMINAL:
                print "    %s of type %s" % (n, self.attribute_types[n])
            else:
                print ("    " + n + " of type nominal with values " +
                       ', '.join(self.attribute_data[n]))
        for d in self.data:
            print d
            
    def append(self, line):
        assert len(line) == len(self.attributes)
        self._parse_data(line)

#a = ArffFile.read('../examples/diabetes.arff')

if __name__ == '__main__':
    a = ArffFile.parse("""% yes
% this is great
@relation foobar
@attribute foo {a,b,c}
@attribute bar real
@data
a, 1
b, 2
c, d
d, 3
""")
    a.dump()
    print a.write()
