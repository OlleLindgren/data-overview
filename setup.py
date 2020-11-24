import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-overview", # Replace with your own username
    version="0.1",
    author="Olle Lindgren",
    author_email="lindgrenolle@live.se",
    description="A package for caching files locally",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OlleLindgren/data-overview",
    packages=setuptools.find_packages(),
    install_requires=[
          'pandas',
          'numpy',
          'termcolor'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)