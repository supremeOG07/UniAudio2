. ./path.sh # set the path

#export CUDA_VISIBLE_DEVICES=0
export HOST_GPU_NUM=4 # set the number of GPU to use
export HOST_NUM=1
export NODE_NUM=1
export INDEX=0
export CHIEF_IP="localhost"
export port=4116
seed=999
learning_rate=1e-5
train_data_path="large_audio.scp"
val_data_path="val_with_duration.scp"
sq_codec_config="ReasoningCodec/sq_config.yaml"
sq_codec_ckpt="ReasoningCodec/ckpt_00615000.pth"
whisper_path="openai/whisper-medium"
transformer_diffusion_config="ReasoningCodec/model_config.json"
llm_path='meta-llama/Llama-3.2-3B'
reason_lm_path='ReasoningCodec/audiothinker.pth'
reconstruction_path='ReasoningCodec/ep1.checkpoint'
prompt_path='prompts/train_prompt.json'
best_rq_ckpt='ReasoningCodec/music_ssl.pt'

NCCL_DEBUG=TRACE python3 -m torch.distributed.run \
    --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
    --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
    train_fsdp.py \
    --exp_dir ./exp \
    --seed $seed \
    --cudnn_deterministic \
    --train_data_path $train_data_path \
    --learning_rate  $learning_rate \
    --val_data_path $val_data_path \
    --learning_rate $learning_rate \
    --sq-config $sq_codec_config \
    --sq-resume $sq_codec_ckpt \
    --whisper_path $whisper_path \
    --reason_lm_path $reason_lm_path \
    --reconstruction_path $reconstruction_path \
    --llm_path $llm_path \
    --prompt_path $prompt_path \
    --best_rq_ckpt $best_rq_ckpt \
    --transformer_diffusion_config $transformer_diffusion_config \
    --mixed-precision 'fp32' \
    --grad-precision 'fp32' \
    --segment_duration 30  \
    --batch_size 16 \
    --print_freq 100 \
    --grad_accum 1 \
    --fine_decoder false \
    --n_epoch 5
