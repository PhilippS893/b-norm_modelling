#!/bin/bash

# this is within model/folder

MODEL_DIR=$1  # which model
METRIC_ID=$2  # which metric
LOOKUP_1=$3   # which keyword1
LOOKUP_2=$4   # which keyword2
OUT_NAME=$5

CROPPED_DIR=${MODEL_DIR}/${METRIC_ID}/cropped
echo "$CROPPED_DIR"
mkdir -p $CROPPED_DIR

crop_images () {
  file=$1
	fName=`basename -- "$file" ".${file##*.}"`
	outName="${CROPPED_DIR}/${fName}.png"

	magick $file -trim +repage $outName
	magick "$outName" -bordercolor white -border 10x10 "$outName"
}

concat_images () {
  files1=$1
  files2=$2
  outName=$3
  magick montage $files1 $files2 -geometry +1+1 -tile 4x2 "${CROPPED_DIR}/${outName}.pdf"
}

DATA_DIR="${MODEL_DIR}/${METRIC_ID}"

# crop all images you can find in the DATA_DIR
for filename in $DATA_DIR/*.png; do
    [ -e "$filename" ] || continue
    crop_images $filename
done

# use this if you want the AgeSex-vs-ThicknessAgeSex images
if test "$METRIC_ID" = "f1" || test "$METRIC_ID" = "f2"
then
  a=`ls $CROPPED_DIR/*_${LOOKUP_1}_thresh*.png`
  b=`ls $CROPPED_DIR/*_${LOOKUP_1}.png`
else
  a=`ls $CROPPED_DIR/*_${LOOKUP_1}_*.png`
  b=`ls $CROPPED_DIR/*_${LOOKUP_2}_*.png`
fi

concat_images "$a" "$b" "$OUT_NAME"
