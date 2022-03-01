# How to create the documentation (Sphinx+GithubPages)

This readme provide a summary for the required steps to build the documentation.

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