#!/bin/bash

BINARIES_DIR=/mnt/md0/flag_detection/binaries
MODEL_DIR=/mnt/md0/flag_detection/models
DATASET_DIR=/mnt/md0/flag_detection/dataset
CMD="python optimization-detector.py"

function extract {
    for ARCH in amd64 aarch64 arm mips powerpc sparc64 riscv64
    do
        mkdir -pv $DATASET_DIR/$ARCH-gcc-o{0,1,2,3,s}-raw
        $CMD extract $BINARIES_DIR/$ARCH-gcc-o0/* $DATASET_DIR/$ARCH-gcc-o0-raw
        $CMD extract $BINARIES_DIR/$ARCH-gcc-o1/* $DATASET_DIR/$ARCH-gcc-o1-raw
        $CMD extract $BINARIES_DIR/$ARCH-gcc-o2/* $DATASET_DIR/$ARCH-gcc-o2-raw
        $CMD extract $BINARIES_DIR/$ARCH-gcc-o3/* $DATASET_DIR/$ARCH-gcc-o3-raw
        $CMD extract $BINARIES_DIR/$ARCH-gcc-os/* $DATASET_DIR/$ARCH-gcc-os-raw
    done

    for ARCH in amd64 aarch64
    do
        mkdir -pv $DATASET_DIR/$ARCH-clang-o{0,1,2,3,s}-raw
        $CMD extract $BINARIES_DIR/$ARCH-clang-o0/* $DATASET_DIR/$ARCH-clang-o0-raw
        $CMD extract $BINARIES_DIR/$ARCH-clang-o1/* $DATASET_DIR/$ARCH-clang-o1-raw
        $CMD extract $BINARIES_DIR/$ARCH-clang-o2/* $DATASET_DIR/$ARCH-clang-o2-raw
        $CMD extract $BINARIES_DIR/$ARCH-clang-o3/* $DATASET_DIR/$ARCH-clang-o3-raw
        $CMD extract $BINARIES_DIR/$ARCH-clang-os/* $DATASET_DIR/$ARCH-clang-os-raw
    done

   # this is very slow and will take several days probably
   # for this reason, in the dataset we put the already extracted data
   #mkdir -pv $DATASET_DIR/$ARCH-{gcc,clang}-o{0,1,2,3,s}-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-gcc-o0 $DATASET_DIR/$ARCH-gcc-o0-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-gcc-o1 $DATASET_DIR/$ARCH-gcc-o1-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-gcc-o2 $DATASET_DIR/$ARCH-gcc-o2-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-gcc-o3 $DATASET_DIR/$ARCH-gcc-o3-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-gcc-os $DATASET_DIR/$ARCH-gcc-os-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-clang-o0 $DATASET_DIR/$ARCH-clang-o0-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-clang-o1 $DATASET_DIR/$ARCH-clang-o1-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-clang-o2 $DATASET_DIR/$ARCH-clang-o2-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-clang-o3 $DATASET_DIR/$ARCH-clang-o3-enc
   #$CMD extract --encoded $BINARIES_DIR/$ARCH-clang-os $DATASET_DIR/$ARCH-clang-os-enc
}

function preprocess {
    for ARCH in amd64 aarch64
    do
      mkdir -pv $MODEL_DIR/$ARCH-mixed-raw{,-no3,-o23}
      echo "$ARCH gcc-O0 + clang-O0"
      $CMD preprocess -c 0 --incomplete $DATASET_DIR/$ARCH-gcc-o0-raw/ $DATASET_DIR/$ARCH-clang-o0-raw/  $MODEL_DIR/$ARCH-mixed-raw
      echo "$ARCH gcc-O1 + clang-O1"
      $CMD preprocess -c 1 --incomplete $DATASET_DIR/$ARCH-gcc-o1-raw/ $DATASET_DIR/$ARCH-clang-o1-raw/  $MODEL_DIR/$ARCH-mixed-raw
      echo "$ARCH gcc-O2 + clang-O2"
      $CMD preprocess -c 2 --incomplete $DATASET_DIR/$ARCH-gcc-o2-raw/ $DATASET_DIR/$ARCH-clang-o2-raw/  $MODEL_DIR/$ARCH-mixed-raw
      cp -v $MODEL_DIR/$ARCH-mixed-raw/{train,test,validate}.bin $MODEL_DIR/$ARCH-mixed-raw-no3/
      echo "$ARCH gcc-O3 + clang-O3"
      $CMD preprocess -c 3 --incomplete $DATASET_DIR/$ARCH-gcc-o3-raw/ $DATASET_DIR/$ARCH-clang-o3-raw/  $MODEL_DIR/$ARCH-mixed-raw
      echo "$ARCH gcc-Os + clang-Os"
      $CMD preprocess -c 4 -s 30219 $DATASET_DIR/$ARCH-gcc-os-raw/ $DATASET_DIR/$ARCH-clang-os-raw/  $MODEL_DIR/$ARCH-mixed-raw
      echo "$ARCH gcc-O3 (no3 dataset) + clang-O3 (no3 dataset)"
      $CMD preprocess -c 2 --incomplete $DATASET_DIR/$ARCH-gcc-o3-raw/  $DATASET_DIR/$ARCH-clang-o3-raw/ $MODEL_DIR/$ARCH-mixed-raw-no3
      echo "$ARCH gcc-Os (no3 dataset) + clang-Os (no3 dataset)"
      $CMD preprocess -c 3 -s 6991 $DATASET_DIR/$ARCH-gcc-os-raw/  $DATASET_DIR/$ARCH-clang-os-raw/ $MODEL_DIR/$ARCH-mixed-raw-no3
      echo "$ARCH gcc-O2 (o23 dataset) + clang-O2 (o23 dataset)"
      $CMD preprocess -c 0 --incomplete $DATASET_DIR/$ARCH-gcc-o2-raw/ $DATASET_DIR/$ARCH-clang-o2-raw/  $MODEL_DIR/$ARCH-mixed-raw-o23
      echo "$ARCH gcc-O3 (o23 dataset) + clang-O3 (o23 dataset)"
      $CMD preprocess -c 1 -s 28422 $DATASET_DIR/$ARCH-gcc-o3-raw/ $DATASET_DIR/$ARCH-clang-o3-raw/  $MODEL_DIR/$ARCH-mixed-raw-o23
      mkdir -v $MODEL_DIR/$ARCH-compiler-raw
      echo "$ARCH gcc-Os + gcc-O1 + gcc-O2 + gcc-O3 + gcc-Os"
      $CMD preprocess -c 0 --incomplete $DATASET_DIR/$ARCH-gcc-o{0,1,2,3,s}-raw/   $MODEL_DIR/$ARCH-compiler-raw
      echo "$ARCH clang-Os + clang-O1 + clang-O2 + clang-O3 + clang-Os"
      $CMD preprocess -c 1 -s 55508 $DATASET_DIR/$ARCH-clang-o{0,1,2,3,s}-raw/ $MODEL_DIR/$ARCH-compiler-raw
    done

    for ARCH in arm mips powerpc sparc64 riscv64
    do
      mkdir -v $MODEL_DIR/$ARCH-gcc-raw
      echo "$ARCH gcc-O0"
      $CMD preprocess -c 0 --incomplete $DATASET_DIR/$ARCH-gcc-o0-raw/ $MODEL_DIR/$ARCH-gcc-raw
      echo "$ARCH gcc-O1"
      $CMD preprocess -c 1 --incomplete $DATASET_DIR/$ARCH-gcc-o1-raw/ $MODEL_DIR/$ARCH-gcc-raw
      echo "$ARCH gcc-O2"
      $CMD preprocess -c 2 --incomplete $DATASET_DIR/$ARCH-gcc-o2-raw/ $MODEL_DIR/$ARCH-gcc-raw
      echo "$ARCH gcc-O3"
      $CMD preprocess -c 3 --incomplete $DATASET_DIR/$ARCH-gcc-o3-raw/ $MODEL_DIR/$ARCH-gcc-raw
      echo "$ARCH gcc-Os"
      $CMD preprocess -c 4 -s 29477 $DATASET_DIR/$ARCH-gcc-os-raw/ $MODEL_DIR/$ARCH-gcc-raw
    done

    mkdir -pv $MODEL_DIR/amd64-mixed-enc
    echo "amd64 gcc-O0 + clang-O0 (encoded)"
    $CMD preprocess --encoded -c 0 --incomplete $DATASET_DIR/amd64-gcc-o0-enc/ $DATASET_DIR/amd64-clang-o0-enc/  $MODEL_DIR/amd64-mixed-enc
    echo "amd64 gcc-O1 + clang-O1 (encoded)"
    $CMD preprocess --encoded -c 1 --incomplete $DATASET_DIR/amd64-gcc-o1-enc/ $DATASET_DIR/amd64-clang-o1-enc/  $MODEL_DIR/amd64-mixed-enc
    echo "amd64 gcc-O2 + clang-O2 (encoded)"
    $CMD preprocess --encoded -c 2 --incomplete $DATASET_DIR/amd64-gcc-o2-enc/ $DATASET_DIR/amd64-clang-o2-enc/  $MODEL_DIR/amd64-mixed-enc
    echo "amd64 gcc-O3 + clang-O3 (encoded)"
    $CMD preprocess --encoded -c 3 --incomplete $DATASET_DIR/amd64-gcc-o3-enc/ $DATASET_DIR/amd64-clang-o3-enc/  $MODEL_DIR/amd64-mixed-enc
    echo "amd64 gcc-Os + clang-Os (encoded)"
    $CMD preprocess --encoded -c 4 -s 23072 $DATASET_DIR/amd64-gcc-os-enc/ $DATASET_DIR/amd64-clang-os-enc/  $MODEL_DIR/amd64-mixed-enc

   mkdir -pv $MODEL_DIR/amd64-compiler-enc
   echo "amd64 gcc-Os + gcc-O1 + gcc-O2 + gcc-O3 + gcc-Os (encoded)"
   $CMD preprocess --encoded -c 0 --incomplete $DATASET_DIR/amd64-gcc-o{0,1,2,3,s}-enc/   $MODEL_DIR/amd64-compiler-enc
   echo "amd64 clang-Os + clang-O1 + clang-O2 + clang-O3 + clang-Os (encoded)"
   $CMD preprocess --encoded -c 1 -s 24090 $DATASET_DIR/amd64-clang-o{0,1,2,3,s}-enc/ $MODEL_DIR/amd64-compiler-enc
}

function train {
    # ETA: ~500 hours, maybe more
    echo "amd64-mixed-raw (CNN)"
    $CMD train -s 26967 -n cnn -b 512 $MODEL_DIR/amd64-mixed-raw
    echo "amd64-mixed-raw (LSTM)"
    $CMD train -s 27958 -n lstm -b 512 $MODEL_DIR/amd64-mixed-raw
    echo "aarch64-mixed-raw (CNN)"
    $CMD train -s 44272 -n cnn -b 512 $MODEL_DIR/aarch64-mixed-raw
    echo "aarch64-mixed-raw (LSTM)"
    $CMD train -s 42463 -n lstm -b 512 $MODEL_DIR/aarch64-mixed-raw
    echo "amd64-mixed-raw-no3 (CNN)"
    $CMD train -s 65189 -n cnn -b 512 $MODEL_DIR/amd64-mixed-raw-no3
    echo "amd64-mixed-raw-no3 (LSTM)"
    $CMD train -s 60663 -n lstm -b 512 $MODEL_DIR/amd64-mixed-raw-no3
    echo "aarch64-mixed-raw-no3 (CNN)"
    $CMD train -s 63435 -n cnn -b 512 $MODEL_DIR/aarch64-mixed-raw-no3
    echo "aarch64-mixed-raw-no3 (LSTM)"
    $CMD train -s 34043 -n lstm -b 512 $MODEL_DIR/aarch64-mixed-raw-no3
    echo "amd64-mixed-raw-o23 (CNN)"
    $CMD train -s 6710 -n cnn -b 512 $MODEL_DIR/amd64-mixed-raw-o23
    echo "amd64-mixed-raw-o23 (LSTM)"
    $CMD train -s 33517 -n lstm -b 512 $MODEL_DIR/amd64-mixed-raw-o23
    echo "aarch64-mixed-raw-o23 (CNN)"
    $CMD train -s 21688 -n cnn -b 512 $MODEL_DIR/aarch64-mixed-raw-o23
    echo "aarch64-mixed-raw-o23 (LSTM)"
    $CMD train -s 10790 -n lstm -b 512 $MODEL_DIR/aarch64-mixed-raw-o23
    echo "amd64-compiler-raw (CNN)"
    $CMD train -s 6627 -n cnn -b 512 $MODEL_DIR/amd64-compiler-raw
    echo "amd64-compiler-raw (LSTM)"
    $CMD train -s 60534 -n lstm -b 512 $MODEL_DIR/amd64-compiler-raw
    echo "aarch64-compiler-raw (CNN)"
    $CMD train -s 17326 -n cnn -b 512 $MODEL_DIR/aarch64-compiler-raw
    echo "aarch64-compiler-raw (LSTM)"
    $CMD train -s 13441 -n lstm -b 512 $MODEL_DIR/aarch64-compiler-raw
    echo "riscv64-gcc-raw (CNN)"
    $CMD train -s 20921 -n cnn -b 512 $MODEL_DIR/riscv64-gcc-raw
    echo "riscv64-gcc-raw (LSTM)"
    $CMD train -s 13297 -n lstm -b 512 $MODEL_DIR/riscv64-gcc-raw
    echo "sparc64-gcc-raw (CNN)"
    $CMD train -s 3970 -n cnn -b 512 $MODEL_DIR/sparc64-gcc-raw
    echo "sparc64-gcc-raw (LSTM)"
    $CMD train -s 11364 -n lstm -b 512 $MODEL_DIR/sparc64-gcc-raw
    echo "powerpc-gcc-raw (CNN)"
    $CMD train -s 31015 -n cnn -b 512 $MODEL_DIR/powerpc-gcc-raw
    echo "powerpc-gcc-raw (LSTM)"
    $CMD train -s 11973 -n lstm -b 512 $MODEL_DIR/powerpc-gcc-raw
    echo "arm-gcc-raw (CNN)"
    $CMD train -s 25850 -n cnn -b 512 $MODEL_DIR/arm-gcc-raw
    echo "arm-gcc-raw (LSTM)"
    $CMD train -s 47545 -n lstm -b 512 $MODEL_DIR/arm-gcc-raw
    echo "mips-gcc-raw (CNN)"
    $CMD train -s 37425 -n cnn -b 512 $MODEL_DIR/mips-gcc-raw
    echo "mips-gcc-raw (LSTM)"
    $CMD train -s 34363 -n lstm -b 512 $MODEL_DIR/mips-gcc-raw
    echo "amd64-mixed-enc (CNN)"
    $CMD train -s 65467 -n cnn -b 512 $MODEL_DIR/amd64-mixed-enc
    echo "amd64-mixed-enc (LSTM)"
    $CMD train -s 63948 -n lstm -b 512 $MODEL_DIR/amd64-mixed-enc
    echo "amd64-compiler-enc (CNN)"
    $CMD train -s 60069 -n cnn -b 512 $MODEL_DIR/amd64-compiler-enc
    echo "amd64-compiler-enc (CNN)"
    $CMD train -s 32750 -n lstm -b 512 $MODEL_DIR/amd64-compiler-enc
}

function evaluate {

     incremental
    for NET in cnn lstm
    do
        BS=512
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5       -o $MODEL_DIR/amd64-mixed-raw/$NET/incremental.csv       $MODEL_DIR/amd64-mixed-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5     -o $MODEL_DIR/aarch64-mixed-raw/$NET/incremental.csv     $MODEL_DIR/aarch64-mixed-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-mixed-raw-no3/$NET/model.h5   -o $MODEL_DIR/amd64-mixed-raw-no3/$NET/incremental.csv   $MODEL_DIR/amd64-mixed-raw-no3
        $CMD evaluate -b $BS -m $MODEL_DIR/aarch64-mixed-raw-no3/$NET/model.h5 -o $MODEL_DIR/aarch64-mixed-raw-no3/$NET/incremental.csv $MODEL_DIR/aarch64-mixed-raw-no3
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-mixed-raw-o23/$NET/model.h5   -o $MODEL_DIR/amd64-mixed-raw-o23/$NET/incremental.csv   $MODEL_DIR/amd64-mixed-raw-o23
        $CMD evaluate -b $BS -m $MODEL_DIR/aarch64-mixed-raw-o23/$NET/model.h5 -o $MODEL_DIR/aarch64-mixed-raw-o23/$NET/incremental.csv $MODEL_DIR/aarch64-mixed-raw-o23
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-compiler-raw/$NET/model.h5    -o $MODEL_DIR/amd64-compiler-raw/$NET/incremental.csv    $MODEL_DIR/amd64-compiler-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/aarch64-compiler-raw/$NET/model.h5  -o $MODEL_DIR/aarch64-compiler-raw/$NET/incremental.csv  $MODEL_DIR/aarch64-compiler-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5       -o $MODEL_DIR/riscv64-gcc-raw/$NET/incremental.csv       $MODEL_DIR/riscv64-gcc-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5       -o $MODEL_DIR/sparc64-gcc-raw/$NET/incremental.csv       $MODEL_DIR/sparc64-gcc-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5       -o $MODEL_DIR/powerpc-gcc-raw/$NET/incremental.csv       $MODEL_DIR/powerpc-gcc-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5           -o $MODEL_DIR/arm-gcc-raw/$NET/incremental.csv           $MODEL_DIR/arm-gcc-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5          -o $MODEL_DIR/mips-gcc-raw/$NET/incremental.csv          $MODEL_DIR/mips-gcc-raw
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-mixed-enc/$NET/model.h5       -o $MODEL_DIR/amd64-mixed-enc/$NET/incremental.csv       $MODEL_DIR/amd64-mixed-enc
        $CMD evaluate -b $BS -m $MODEL_DIR/amd64-compiler-enc/$NET/model.h5    -o $MODEL_DIR/amd64-compiler-enc/$NET/incremental.csv    $MODEL_DIR/amd64-compiler-enc
    done

    confusion
    for NET in cnn lstm
    do
        BS=512
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5       -o $MODEL_DIR/amd64-mixed-raw/$NET/confusion.txt       $MODEL_DIR/amd64-mixed-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5     -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusion.txt     $MODEL_DIR/aarch64-mixed-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-mixed-raw-no3/$NET/model.h5   -o $MODEL_DIR/amd64-mixed-raw-no3/$NET/confusion.txt   $MODEL_DIR/amd64-mixed-raw-no3
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/aarch64-mixed-raw-no3/$NET/model.h5 -o $MODEL_DIR/aarch64-mixed-raw-no3/$NET/confusion.txt $MODEL_DIR/aarch64-mixed-raw-no3
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-mixed-raw-o23/$NET/model.h5   -o $MODEL_DIR/amd64-mixed-raw-o23/$NET/confusion.txt   $MODEL_DIR/amd64-mixed-raw-o23
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/aarch64-mixed-raw-o23/$NET/model.h5 -o $MODEL_DIR/aarch64-mixed-raw-o23/$NET/confusion.txt $MODEL_DIR/aarch64-mixed-raw-o23
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-compiler-raw/$NET/model.h5    -o $MODEL_DIR/amd64-compiler-raw/$NET/confusion.txt    $MODEL_DIR/amd64-compiler-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/aarch64-compiler-raw/$NET/model.h5  -o $MODEL_DIR/aarch64-compiler-raw/$NET/confusion.txt  $MODEL_DIR/aarch64-compiler-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5       -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusion.txt       $MODEL_DIR/riscv64-gcc-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5       -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusion.txt       $MODEL_DIR/sparc64-gcc-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5       -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusion.txt       $MODEL_DIR/powerpc-gcc-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5           -o $MODEL_DIR/arm-gcc-raw/$NET/confusion.txt           $MODEL_DIR/arm-gcc-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5          -o $MODEL_DIR/mips-gcc-raw/$NET/confusion.txt          $MODEL_DIR/mips-gcc-raw
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-mixed-enc/$NET/model.h5       -o $MODEL_DIR/amd64-mixed-enc/$NET/confusion.txt       $MODEL_DIR/amd64-mixed-enc
        $CMD evaluate -b $BS -c 2048 -m $MODEL_DIR/amd64-compiler-enc/$NET/model.h5    -o $MODEL_DIR/amd64-compiler-enc/$NET/confusion.txt    $MODEL_DIR/amd64-compiler-enc
    done

    BS=512
    NET=lstm
    $CMD evaluate -b $BS -c  65 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/amd64-mixed-raw/$NET/confusionO0.txt    $MODEL_DIR/amd64-mixed-raw
    $CMD evaluate -b $BS -c 223 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/amd64-mixed-raw/$NET/confusionO1.txt    $MODEL_DIR/amd64-mixed-raw
    $CMD evaluate -b $BS -c 220 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/amd64-mixed-raw/$NET/confusionO2.txt    $MODEL_DIR/amd64-mixed-raw
    $CMD evaluate -b $BS -c 257 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/amd64-mixed-raw/$NET/confusionO3.txt    $MODEL_DIR/amd64-mixed-raw
    $CMD evaluate -b $BS -c 125 -m $MODEL_DIR/amd64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/amd64-mixed-raw/$NET/confusionOS.txt    $MODEL_DIR/amd64-mixed-raw
    $CMD evaluate -b $BS -c  68 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusionO0.txt    $MODEL_DIR/aarch64-mixed-raw
    $CMD evaluate -b $BS -c 232 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusionO1.txt    $MODEL_DIR/aarch64-mixed-raw
    $CMD evaluate -b $BS -c 220 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusionO2.txt    $MODEL_DIR/aarch64-mixed-raw
    $CMD evaluate -b $BS -c 260 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusionO3.txt    $MODEL_DIR/aarch64-mixed-raw
    $CMD evaluate -b $BS -c 132 -m $MODEL_DIR/aarch64-mixed-raw/$NET/model.h5    -o $MODEL_DIR/aarch64-mixed-raw/$NET/confusionOS.txt    $MODEL_DIR/aarch64-mixed-raw
    $CMD evaluate -b $BS -c  84 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusionO0.txt    $MODEL_DIR/riscv64-gcc-raw
    $CMD evaluate -b $BS -c 160 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusionO1.txt    $MODEL_DIR/riscv64-gcc-raw
    $CMD evaluate -b $BS -c 166 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusionO2.txt    $MODEL_DIR/riscv64-gcc-raw
    $CMD evaluate -b $BS -c 218 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusionO3.txt    $MODEL_DIR/riscv64-gcc-raw
    $CMD evaluate -b $BS -c 113 -m $MODEL_DIR/riscv64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/riscv64-gcc-raw/$NET/confusionOS.txt    $MODEL_DIR/riscv64-gcc-raw
    $CMD evaluate -b $BS -c 104 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusionO0.txt    $MODEL_DIR/sparc64-gcc-raw
    $CMD evaluate -b $BS -c 192 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusionO1.txt    $MODEL_DIR/sparc64-gcc-raw
    $CMD evaluate -b $BS -c 220 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusionO2.txt    $MODEL_DIR/sparc64-gcc-raw
    $CMD evaluate -b $BS -c 268 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusionO3.txt    $MODEL_DIR/sparc64-gcc-raw
    $CMD evaluate -b $BS -c 132 -m $MODEL_DIR/sparc64-gcc-raw/$NET/model.h5    -o $MODEL_DIR/sparc64-gcc-raw/$NET/confusionOS.txt    $MODEL_DIR/sparc64-gcc-raw
    $CMD evaluate -b $BS -c 136 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5    -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusionO0.txt    $MODEL_DIR/powerpc-gcc-raw
    $CMD evaluate -b $BS -c 224 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5    -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusionO1.txt    $MODEL_DIR/powerpc-gcc-raw
    $CMD evaluate -b $BS -c 264 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5    -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusionO2.txt    $MODEL_DIR/powerpc-gcc-raw
    $CMD evaluate -b $BS -c 292 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5    -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusionO3.txt    $MODEL_DIR/powerpc-gcc-raw
    $CMD evaluate -b $BS -c 148 -m $MODEL_DIR/powerpc-gcc-raw/$NET/model.h5    -o $MODEL_DIR/powerpc-gcc-raw/$NET/confusionOS.txt    $MODEL_DIR/powerpc-gcc-raw
    $CMD evaluate -b $BS -c 184 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5    -o $MODEL_DIR/mips-gcc-raw/$NET/confusionO0.txt    $MODEL_DIR/mips-gcc-raw
    $CMD evaluate -b $BS -c 276 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5    -o $MODEL_DIR/mips-gcc-raw/$NET/confusionO1.txt    $MODEL_DIR/mips-gcc-raw
    $CMD evaluate -b $BS -c 284 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5    -o $MODEL_DIR/mips-gcc-raw/$NET/confusionO2.txt    $MODEL_DIR/mips-gcc-raw
    $CMD evaluate -b $BS -c 376 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5    -o $MODEL_DIR/mips-gcc-raw/$NET/confusionO3.txt    $MODEL_DIR/mips-gcc-raw
    $CMD evaluate -b $BS -c 188 -m $MODEL_DIR/mips-gcc-raw/$NET/model.h5    -o $MODEL_DIR/mips-gcc-raw/$NET/confusionOS.txt    $MODEL_DIR/mips-gcc-raw
    $CMD evaluate -b $BS -c  72 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5    -o $MODEL_DIR/arm-gcc-raw/$NET/confusionO0.txt    $MODEL_DIR/arm-gcc-raw
    $CMD evaluate -b $BS -c 188 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5    -o $MODEL_DIR/arm-gcc-raw/$NET/confusionO1.txt    $MODEL_DIR/arm-gcc-raw
    $CMD evaluate -b $BS -c 174 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5    -o $MODEL_DIR/arm-gcc-raw/$NET/confusionO2.txt    $MODEL_DIR/arm-gcc-raw
    $CMD evaluate -b $BS -c 232 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5    -o $MODEL_DIR/arm-gcc-raw/$NET/confusionO3.txt    $MODEL_DIR/arm-gcc-raw
    $CMD evaluate -b $BS -c 106 -m $MODEL_DIR/arm-gcc-raw/$NET/model.h5    -o $MODEL_DIR/arm-gcc-raw/$NET/confusionOS.txt    $MODEL_DIR/arm-gcc-raw

    for COMPILER in compiler
    do
      $CMD infer -b 512 -m $MODEL_DIR/amd64-$COMPILER-raw/lstm/model.h5 -o $MODEL_DIR/amd64-$COMPILER-raw/lstm/ubuntu_report.txt /mnt/md0/flag_detection/stats/linux/bins/*
      $CMD infer -b 512 -m $MODEL_DIR/amd64-$COMPILER-raw/lstm/model.h5 -o $MODEL_DIR/amd64-$COMPILER-raw/lstm/ubuntu_report.txt /mnt/md0/flag_detection/stats/linux/libs/*
      $CMD infer -b 512 -m $MODEL_DIR/amd64-$COMPILER-raw/lstm/model.h5 -o $MODEL_DIR/amd64-$COMPILER-raw/lstm/macos_report.txt /mnt/md0/flag_detection/stats/macos/bins/*
      $CMD infer -b 512 -m $MODEL_DIR/amd64-$COMPILER-raw/lstm/model.h5 -o $MODEL_DIR/amd64-$COMPILER-raw/lstm/macos_report.txt /mnt/md0/flag_detection/stats/macos/libs/*
    done
}

extract
preprocess
train
evaluate
