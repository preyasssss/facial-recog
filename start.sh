#!/bin/bash
echo "Compilare..."
./compilare.sh
if [ $? -ne 0 ]; then
    echo "Eroare la compilare!"
    exit 1
fi
echo "Lansare aplicație..."
OPENCV_JAR="lib/opencv-4100.jar"
OUT_DIR="out"
java -cp "$OUT_DIR:$OPENCV_JAR" -Djava.library.path="lib" Main --ui