from setuptools import setup, find_packages


# description
with open ('README.md', 'r') as fh:
	long_description = fh.read()

desc = "Package by the Data-Scientists for the Data Scientists ; with Scikit-learn type fit() transform() functionality"
URL = "https://github.com/samarth-agrawal-86/fast_ml"
# desc = 'This package is like having a junior Data Scientist working for you. So that you can delegate a lot of work and you focus on bringing insights. Techniques for making machine learning easy',

setup(
		name = 'fast_ml',
		version = 3.39,
		description = desc,
		long_description = long_description,
		long_description_content_type = "text/markdown",	
		url = URL,
		packages=find_packages(include=['fast_ml', 'fast_ml.*']),
		author = 'Samarth Agrawal',
		author_email = 'samarth.agrawal.86@gmail.com',
		python_requires='>=3.6',
		zip_safe = False
	)