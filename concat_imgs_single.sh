#!/bin/bash

MODEL_DIR=$1  # which model
METRIC_ID=$2  # which metric
LOOKUP_1=$3   # which keyword1
OUT_NAME=$4
ORIENTATION=$5

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
  outName=$2
  #magick montage $files1 -geometry +1+1 -tile 4x1 "${CROPPED_DIR}/${outName}.png"
  magick montage $files1 -geometry +1+1 -tile $ORIENTATION "${CROPPED_DIR}/${outName}.pdf"
}

DATA_DIR="${MODEL_DIR}/${METRIC_ID}"

# crop all images you can find in the DATA_DIR
for filename in $DATA_DIR/*.png; do
    [ -e "$filename" ] || continue
    crop_images $filename
done

a=`ls $CROPPED_DIR/*_${LOOKUP_1}_*.png`

concat_images "$a" "$OUT_NAME"