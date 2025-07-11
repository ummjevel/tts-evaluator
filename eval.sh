#!/bin/bash

# 변수 선언
GEN_DIR="/data/voice/tts/zoey/aicess-cosyvoice/examples/libritts/cosyvoice2/exp/cosyvoice2/kss_24k_avg/test_folder"
REF_TEXTS="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/transcript_original_expand_val_short500.txt"
SAVE_PATH="./result/test-summary_result.json"
SAVE_ASR_JSON_PATH="./result/test-whisper_result.json"

# 실행
python main.py \
    --gen_dir "$GEN_DIR" \
    --ref_texts "$REF_TEXTS" \
    --save_path "$SAVE_PATH" \
    --save_asr_json_path "$SAVE_ASR_JSON_PATH"


# 변수 선언
GEN_DIR="/data/voice/tts/zoey/aicess-cosyvoice/examples/libritts/cosyvoice2/exp/cosyvoice2/kss_24k_avg/batch_infer_sft"
REF_TEXTS="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/transcript_original_expand_val_short500.txt"
SAVE_PATH="./result/kss_24k_avg-batch_infer_sft-summary_result.json"
SAVE_ASR_JSON_PATH="./result/kss_24k_avg-batch_infer_sft-whisper_result.json"

# 실행
python main.py \
    --gen_dir "$GEN_DIR" \
    --ref_texts "$REF_TEXTS" \
    --save_path "$SAVE_PATH" \
    --save_asr_json_path "$SAVE_ASR_JSON_PATH"

# 변수 선언
GEN_DIR="/data/voice/tts/zoey/aicess-cosyvoice/examples/libritts/cosyvoice2/exp/cosyvoice2/kss_24k_avg_sft/batch_infer_sft"
REF_TEXTS="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/transcript_original_expand_val_short500.txt"
SAVE_PATH="./result/kss_24k_avg_sft-batch_infer_sft-summary_result.json"
SAVE_ASR_JSON_PATH="./result/kss_24k_avg_sft-batch_infer_sft-whisper_result.json"

# 실행
python main.py \
    --gen_dir "$GEN_DIR" \
    --ref_texts "$REF_TEXTS" \
    --save_path "$SAVE_PATH" \
    --save_asr_json_path "$SAVE_ASR_JSON_PATH"


# 변수 선언
GEN_DIR="/data/voice/tts/zoey/aicess-cosyvoice/examples/libritts/cosyvoice2/exp/cosyvoice2/kss_24k_avg_sft/batch_infer_sft_woprompt"
REF_TEXTS="/data/voice/dataset/aihub/138.뉴스_대본_및_앵커_음성_데이터/transcript_original_expand_val_short500.txt"
SAVE_PATH="./result/kss_24k_avg_sft-batch_infer_sft_woprompt-summary_result.json"
SAVE_ASR_JSON_PATH="./result/kss_24k_avg_sft-batch_infer_sft_woprompt-whisper_result.json"

# 실행
python main.py \
    --gen_dir "$GEN_DIR" \
    --ref_texts "$REF_TEXTS" \
    --save_path "$SAVE_PATH" \
    --save_asr_json_path "$SAVE_ASR_JSON_PATH"
