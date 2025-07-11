#!/bin/bash

# 변수 선언
GEN_DIR="/data/voice/tts/zoey/train/aicess-cosyvoice/examples/libritts/cosyvoice2/exp/cosyvoice2/aicc_spk066_sft_avg/batch_infer_sft"
REF_DIR="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/wav_24k/SPK066"
REF_TEXTS="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/transcript_original_expand_val_aicc3.txt"
SAVE_PATH="./result/aicc_spk066_sft_avg-batch_infer_sft-summary_result.json"
SAVE_ASR_JSON_PATH="./result/aicc_spk066_sft_avg-batch_infer_sft-whisper_result.json"

# 실행
python main_batch.py \
    --gen_dir "$GEN_DIR" \
    --ref_texts "$REF_TEXTS" \
    --save_path "$SAVE_PATH" \
    --save_asr_json_path "$SAVE_ASR_JSON_PATH" \
    --device 0



