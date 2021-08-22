import setuptools


def get_data_files(kind: str) -> list:
    from os import listdir
    from os.path import isfile, join, dirname, realpath
    dir_path = dirname(realpath(__file__))

    if kind == 'data':
        data_path = join(dir_path, 'bclm', 'data')
    elif kind == 'yap':
        data_path = join(dir_path, 'bclm', 'data', 'yap_outputs')
    else:
        raise ValueError("Only support [data|yap] as an input")
    return [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]


with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='bclm',
        version='1.0.1',
        author="Dan Bareket",
        author_email="dbareket@gmail.com",
        description="THE go-to place for all Python Hebrew Treebank processing tasks.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/OnlpLab/bclm",
        download_url="https://github.com/OnlpLab/bclm/archive/refs/tags/v1.0.1-beta.tar.gz",
        packages=['bclm', 'bclm/data'],
        package_data={'bclm': get_data_files('data'), 'bclm/data': get_data_files('yap')},
        install_requires=['pandas',
                          'conllu',
                          'numpy'],
        classifiers=["Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent", ],
    )
