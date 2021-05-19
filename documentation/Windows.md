# Tips for Windows User

The code is mostly developed on Linux and therefore there can be unkown bugs in Windows, but in general it should be working on both systems. If you have problems, please report them.

Anaconda helps to deal with environments on Windows: https://www.anaconda.com/products/individual
You can lunch PyCharm from there.

## Run make test on Windows terminal

The make scripts are a powerful tool and should be working. Especially so you can `make test` your work.

Make is kinda difficult to run on Windows. There seem to be some but not good documented/general solutions. If you can improve this, please help.

### Windows Subsystems for Linux

- [Install the windows subsystem for linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) 
- After enabling WSL2, install ubuntu from Microsoft store.
- [Enable the Hyper-V](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v)
- Open the ubuntu terminal and install python3
  
  - Use `python3 –version` to verify python3 is installed.
- Open pycharm and follow the steps below:
  1.	Go to File->Settings->Tools->Terminal
  2.	In application setting select the shell path for the ubuntu.
  3.	Default path will be: C:\Users\name\AppData\Local\Microsoft\WindowsApps\ubuntu2004.exe
  To reach to this path you must select "Show hidden files and directories".
  4.	Now you have got an ubuntu terminal
  - In ubuntu terminal select the directory to be SMIDA  (`/mnt/c/...` to access windows folders)
  - Install all the required packages like pandas using command `sudo apt-get install python3-pandas`. Install other packages in same way.
  - Install make package using `sudo apt-get install make`
- Now you can run make test in terminal
