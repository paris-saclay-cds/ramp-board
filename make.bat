@ECHO OFF

REM command file for RAMP install in Windows

set PYTHON=python
set PYTEST=pytest

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  install   install the RAMP package into the current environment
	echo.  inplace   install the RAMP package in developer mode
	echo.  test      test the developer version
	echo.  clean     clean the compiled file
	goto end
)

if "%1" == "install" (
    cd ramp-frontend && pip install . && cd ..
	cd ramp-database && pip install . && cd ..
	cd ramp-engine && pip install . && cd ..
	cd ramp-utils && pip install . && cd ..
)

if "%1" == "inplace" (
	cd ramp-frontend && pip install -e . && cd ..
	cd ramp-database && pip install -e . && cd ..
	cd ramp-engine && pip install -e . && cd ..
	cd ramp-utils && pip install -e . && cd ..
)

if "%1" == "test"(
    %PYTEST% -vsl .
)

if "%1" == "clean"(
    cd ramp-frontend && %PYTHON% setup.py clean && cd ..
	cd ramp-database && %PYTHON% setup.py clean && cd ..
	cd ramp-engine && %PYTHON% setup.py clean && cd ..
	cd ramp-utils && %PYTHON% setup.py clean && cd ..
)