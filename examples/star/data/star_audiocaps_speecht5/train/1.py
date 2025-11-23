import json, sys

# 读 B，收集 audio_id
b_ids = set()
with open("/hpc_stor03/sjtu_home/yixuan.li/work/audio_embeds/x_to_audio_generation/data/star_audiocaps_qformer/train/audio10.jsonl") as f:
    for line in f:
        if line.strip():
            b_ids.add(json.loads(line)["audio_id"])

# 过滤 A 并写 C
with open("/hpc_stor03/sjtu_home/yixuan.li/work/audio_embeds/x_to_audio_generation/data/star_audiocaps_qformer/train/caption.jsonl") as fin, open("/hpc_stor03/sjtu_home/yixuan.li/work/audio_embeds/x_to_audio_generation/data/star_audiocaps_qformer/train/caption10.jsonl", "w") as fout:
    for line in fin:
        if line.strip():
            j = json.loads(line)
            if j["audio_id"] in b_ids:
                fout.write(line)