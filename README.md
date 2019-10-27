# ExoBoost
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/rndev2017/ExoBoost/graphs/commit-activity)
![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)
[![nasa](https://img.shields.io/badge/powered%20by-NASA%20Exoplanet%20Archive-blue)](https://exoplanetarchive.ipac.caltech.edu/)<br>
Using existing radial velocity data to identify exoplanet companions with XGBoost <br>

### Code Author
Rohan S. Nagavardhan: [@rndev2017](https://github.com/rndev2017) <br>

### Directories
[src/](https://github.com/rndev2017/ExoBoost/tree/master/src)
- contains the XGBoost Classifier
- contains the script used to label my data set
- contains the `Star` class
    - plot Radial Velocity Plots
    - plot phased Radial Velocity Plots
    - plot periodograms
    - fit phased radial velocity curves

[Research Paper/](https://github.com/rndev2017/ExoBoost/tree/master/Research%20Paper)
- My research manuscript anticipating submission to the Regeneron STS Competition
- <strong>Title</strong>: Identifying Exoplanet Companions using Radial Velocity with Deep Learning

[data/](https://github.com/rndev2017/ExoBoost/tree/master/data)
- contains the data used to train the XGBoost Classifier
- formatted in `.csv` format

## Dependencies
- XGBoost ([instructions](https://xgboost.readthedocs.io/en/latest/build.html#python-package-installation))
- Pandas ([instructions](https://pandas.pydata.org/pandas-docs/stable/install.html))
- NumPy ([instructions](https://scipy.org/install.html))
- SciPy ([instructions](https://scipy.org/install.html))
- Astropy ([instructions](https://www.astropy.org/))
- Matplotlib ([instructions](https://matplotlib.org/users/installing.html))
- Scikit-Learn - ([instructions](https://scikit-learn.org/stable/install.html))
- Seaborn - ([instructions](https://seaborn.pydata.org/installing.html))
- Astroquery - ([instructions](https://astroquery.readthedocs.io/en/latest/#installation))

# Future Plans
I will be submitting this work to a conference, [Exoplanets in our Backyard](https://www.hou.usra.edu/meetings/exoplanets2020/), in February of 2020. I might be given the oppurtunity to present this work at [LISEF](https://www.lisef.org/) and various science research fairs in the near future as part of my science research class. Additionally, I am considering submitting this work to a journal for publication. As for the project itself, I will be working on improving the model by creating a simulated dataset to add to the existing dataset of real exoplanet data. Furthermore, I am thinking about using a transfer learning approach by repurposing [AstroNet](https://github.com/google-research/exoplanet-ml) for exoplanet companion identification with radial velocity. Keep following this directory for updates on this project.
