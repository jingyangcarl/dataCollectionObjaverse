export BLENDER_PATH=/home/jyang/projects/dataCollectionObjaverse/blender-3.6.23-linux-x64/blender
export PIPELINE_PATH=/home/jyang/projects/dataCollectionObjaverse/combined_pipeline.py
# export INPUT_DIR=/home/jyang/projects/dataCollectionObjaverse/assets/blenderkit
export INPUT_DIR=/home/jyang/projects/dataCollectionObjaverse/assets/fbx
# export INPUT_DIR=/home/jyang/projects/dataCollectionObjaverse/assets/fbx_debug
export OUTPUT_DIR=/home/jyang/projects/dataCollectionObjaverse/renderings/output_fix_lighting
export SHARDS_ROOT=/home/jyang/projects/dataCollectionObjaverse/renderings/shards

# when using eevee, gpus will only contribute the # of processes
# python shard_and_launch.py \
#     --input_dir $INPUT_DIR \
#     --output_dir $OUTPUT_DIR \
#     --blender $BLENDER_PATH \
#     --pipeline $PIPELINE_PATH \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --shards_root $SHARDS_ROOT
# exit 0

# test run and debug
export OUTPUT_DIR=/home/jyang/projects/dataCollectionObjaverse/renderings/output_dir_noshard
$BLENDER_PATH --background --python $PIPELINE_PATH