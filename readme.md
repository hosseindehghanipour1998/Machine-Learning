[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Introduction to Jupyter Notebook


## Build a Conda Environment
Our default _base_ is where we have installed _Anaconda_.

- Go to the directory you wish to create your environment using _Terminal_.
  - `cd <YourPath>`
  - `conda create --prefix ./env <tools You need>`
  - `conda install <your tool>`

As an Example :
```
$ cd /d G:\GitRepositories\NewEnvironment
$ conda create --prefix ./env pandas matplotlib numpy scikit-Learn
$ conda activate
$ conda install jupyter

```

## How To Open Jupyter Notebook
- Firstly you should build a `Conda Environment`
- Open `Terminal`
- `cd` to the directory that contains the built _environment_.

```
cd /d <path>
Ex : cd /d G:\GitRepository\Machine-Learning\
```
<<<<<<< HEAD
- Activate _Anaconda Environment_ simply by typing : `$ conda activate [environmentName]`
  - If you don't specifically name the `[environmentName]` name , it will automatically open the base environment.
=======
- Acitvate _Anaconda Environment_ simply by typing : `$ conda activate [environmentName]`
  - If you don't specifictly name the `[environmentName]` name , it will automatically open the base environment.
  - You can see the existing environments by simply typing `conda info --env`
>>>>>>> 9709b90ad3bd01429af240a6e5528dee76060347
- Open Jupyter Notebook : `$ jupyter notebook`

As an Example :
```
$ cd /d G:\GitRepository\Machine-Learning/
$ conda activate RegEnv
$ jupyter notebook
```


## Summary
```
$ cd /d G:\GitRepositories\NewEnvironment
$ conda create --prefix ./env pandas matplotlib numpy scikit-Learn
$ conda activate
$ conda install jupyter
$ jupyter notebook
```


<hr>



## Jupyter Guide

##### Switch to Markdown

  - Press `Esc`
  - Press `m` on the keyboard

##### Switch to Code

- Press `Esc`
- Press `y` on the keyboard

##### Delete a Line

- Press `Esc`
- Press `x` on the keyboard


##### Run a Line

- `Shift + Enter`

##### Guide of each function
- Go to the _paranthesis_ `()` of the function
- push `shift + tab` to see the function's documentation

##### Auto Complete
- Only `Tab` ...  push multiple times to show the options.



<hr>


# License
This repository is released under [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). To put it in a nutshell, This means that you can use the source codes in any opensource projects or contributions. Attribution is not mandatory but appreciated.

***To remove all barriers in the way of science.***
