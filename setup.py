from setuptools import setup

setup(
    name='mmot',
    version='0.1.0',    
    description='A package for solving MMOT problems',
    url='https://github.com/simda-muri/mmot',
    author='Matthew Parno and Bohan Zhou',
    author_email='matthew.d.parno@dartmouth.edu',
    license='BSD #-clause',
    packages=['mmot'],
    install_requires=['scipy',
                      'numpy',
		      'matplotlib',
                      'cairocffi',
                      'igraph',
                      'w2 @ git+https://github.com/Math-Jacobs/bfm@main#egg=w2&subdirectory=python'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3',
    ],
)
