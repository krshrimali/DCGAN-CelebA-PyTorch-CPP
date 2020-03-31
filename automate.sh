#! /bin/sh

# Variables for folder names and paths 
BUILD_F=build
OUTPUT_F=build/output

PATH_LIBTORCH_DIR=/root/libtorch
PATH_LIBTORCH_BUILD=/root//libtorch/share/cmake/Torch/



if [ -d $BUILD_F ] ; then
	echo "build folder Already Present... !! skipping folder creation"
else
	echo "Creating build folder"
	mkdir build && echo "Done" || (echo "Folder creation failed.. exiting..." && exit 0)
fi


echo "Running Cmake inside the build folder..."
(cd build && cmake -DCMAKE_PREFIX_PATH=${PATH_LIBTORCH_DIR} -DCMAKE_MODULE_PATH=${PATH_LIBTORCH_BUILD} ..) || (echo "${bold}${red}cmake command failed... exiting.." && exit 0)


if [ -d ${OUTPUT_F} ] ; then
	echo "output folder already exists inside build...!! skipping folder creation"
else
	echo "Creating output folder inside build"
	mkdir -p build/output && echo "Done" || (echo "Folder creation failed.. exiting" && exit 0)
fi

echo "Running the make command"

(cd build && make) && echo "Done" || (echo "make command failed.. exiting ..." && exit 0)
