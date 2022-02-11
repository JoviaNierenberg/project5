from setuptools import setup, find_packages

setup(
    name= 'kmeans_and_silhouette',
    version= '0.1.0',
    author= 'Some Developer',
    author_email= 'yellatme@somewhere.net',
    packages= [
		find_packages('cluster', 'cluster.*')
		],
    description= 'kmeans',
	install_requires= [
		'numpy',
    	'scipy',
    	'pytest'
		]
)