dirlist="/home/mgallet/Documents/DATA/BumbleBuzz"

# iterate over all subdirectories in $dirlist
for fullpath in "$dirlist"/*/ ; do
	[ -d "$fullpath" ] || continue
	curdir=$(basename "$fullpath")
	# skip the predictions output folder itself if present
	if [ "$curdir" = "detection_MobileNetV2" ]; then
		continue
	fi

	# check indices file exists before running eval
	indices_file="$dirlist/detection_MobileNetV2/indices_${curdir}.csv"
	if [ ! -f "$indices_file" ]; then
		echo "Warning: indices file not found for site $curdir — expected $indices_file; skipping"
		continue
	fi

	python evaluation/eval.py \
		--dir "$dirlist" \
		--name "$curdir" \
		--pred detection_MobileNetV2
done

# Create prediction and evaluation directories
mkdir -p "$dirlist/detection_MobileNetV2/prediction"
mkdir -p "$dirlist/detection_MobileNetV2/evaluation"

# Move indices_*.csv files to prediction folder
for indices_file in "$dirlist/detection_MobileNetV2"/indices_*.csv; do
	if [ -f "$indices_file" ]; then
		mv "$indices_file" "$dirlist/detection_MobileNetV2/prediction/"
	fi
done

# Move all other CSV and PNG files to evaluation folder
for file in "$dirlist/detection_MobileNetV2"/*.csv "$dirlist/detection_MobileNetV2"/*.png; do
	if [ -f "$file" ]; then
		mv "$file" "$dirlist/detection_MobileNetV2/evaluation/"
	fi
done


