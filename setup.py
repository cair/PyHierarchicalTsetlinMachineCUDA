from setuptools import *

setup(
   name='PyHierarchicalTsetlinMachineCUDA',
   version='0.2.6',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='https://github.com/cair/PyHierarchicalTsetlinMachineCUDA/',
   license='MIT',
   description='Hierarchical Tsetlin Machine Architecture.',
   long_description='Hierarchical Tsetlin Machine Architecture.',
   keywords ='pattern-recognition cuda machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic tsetlin-machine regression convolution classification multi-layer',
   packages=['PyHierarchicalTsetlinMachineCUDA'],
   install_requires=[
		'numpy',
		'pycuda',
		'scipy',
		'scikit-learn',
      'scikit-image',
		'networkx',
	],
	extras_require={
		'examples': ['tensorflow', 'matplotlib', 'pygraphviz'],
	}
)
