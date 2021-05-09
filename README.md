# bclm
[![CC BY 4.0][cc-by-shield]][cc-by]

THE go-to place for all Python Hebrew Treebank processing tasks.

## Installation

The installation is standard:

pip instatll <PATH_TO_WHEEL>

In order to create the wheel:
1. Make sure you have the latest versions of setuptools and wheel installed:
`python3 -m pip install --user --upgrade setuptools wheel`
2. Now run this command from the same directory where setup.py is located:
`python setup.py bdist_wheel`
3. It will generate a wheel file saved in the `dist` folder. 
4. You can now run `pip install <PATH_TO_WHEEL>`

## Citation

Please cite the NEMO^2 paper if you use bclm in your research:
```bibtex
@article{DBLP:journals/corr/abs-2007-15620,
  author    = {Dan Bareket and
               Reut Tsarfaty},
  title     = {Neural Modeling for Named Entities and Morphology (NEMO{\^{}}2)},
  journal   = {CoRR},
  volume    = {abs/2007.15620},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.15620},
  archivePrefix = {arXiv},
  eprint    = {2007.15620},
  timestamp = {Mon, 03 Aug 2020 14:32:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2007-15620.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
