#!/bin/bash
OPENCV_JAR="lib/opencv-4100.jar"
SRC_DIR="src"
OUT_DIR="out"
rm -rf $OUT_DIR
mkdir $OUT_DIR
javac -cp $OPENCV_JAR -d $OUT_DIR $(find $SRC_DIR -name "*.java")
echo "Compilare finalizată."