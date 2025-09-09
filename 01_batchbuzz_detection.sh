dirlist="/home/mgallet/Documents/DATA/BumbleBuzz"

# iterate over all subdirectories in $dirlist
for fullpath in "$dirlist"/*/ ; do
	[ -d "$fullpath" ] || continue
	curdir=$(basename "$fullpath")
	python process.py \
		--save_audio_flac 1 \
		--data_path "$fullpath" \
		--name "$curdir" \
		--l 5 \
		--save_path "$dirlist/detection_MobileNetV2" \
		--model_type MobileNetV2 \
		--audio_format wav
done

# Create data_cut directory and move audio_* folders there
mkdir -p "$dirlist/detection_MobileNetV2/data_cut"
for audio_dir in "$dirlist/detection_MobileNetV2"/audio_*; do
    if [ -d "$audio_dir" ]; then
        mv "$audio_dir" "$dirlist/detection_MobileNetV2/data_cut/"
    fi
done

