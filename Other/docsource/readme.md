# How to create the documentation (Sphinx+GithubPages)

This readme provide a summary for the required steps to build the documentation.

## Installation of Sphinx and needed packages

On Linux run

~~~bash
sudo apt install python3-sphinx -y
~~~

On a mac do

~~~bash
brew install sphinx-doc
~~~

Wee need a few packages. To get them run

~~~bash
pip install sphinxawesome_theme myst_parser sphinx_press_theme
~~~

## Re-building the doc after code changes

Remove the contents of `docsource/source` to avoid problems if filenames have changed.
Also remove `docs/doctrees` and `docs/html`.

Use a terminal to go into the `docsource` folder. Then run

~~~bash
sphinx-apidoc -f -o source/ ../src/quocslib/
~~~

This will create the source files.

Then to build the documentation do

~~~bash
make github
~~~

also in the `docsource` folder.


## Building from scratch

* Create folder `docsource/`. In this folder:
    - Use the command `sphinx-quickstart` to generate automatically the `makefile` and `conf.py`.
        * Modify `conf.py` in order to change style etc...
        * Change `BUILDDIR=../docs/` in `makefile`.
    - Use the command `sphinx-apidoc -f -o source/ ../src/quocslib/` to generate the folder `source/`. The latter contains the part of the documentation that can be built automatically from the code.
    - Create folder `chapters/`. This folder will contain the part of the documentation that is written manually, like user guide and tutorials.
    - Use `make` command. This will generate the html file inside the folder `../docs/`.

* Go into folder `docs/`. 
    - In order to make **GithubPages** working correctly the html files should be on the top level of this folder. At the moment they are not. To solve this problem, create a file `index.html` which redirect to `html/index.html`.
    - If you want to use the theme `sphinx` add `.nojekyll` file into this folder.

With these steps the documentation inside **GithubPages** should work.