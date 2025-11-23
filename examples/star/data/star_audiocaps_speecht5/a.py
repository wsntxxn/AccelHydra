import json
import os

def modify_jsonl_speech_prefix(input_file, output_file, new_prefix):
    """
    修改JSONL文件中speech字段的路径前缀
    
    参数:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        new_prefix: 新的路径前缀（例如：/new/path/to/features/）
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # 解析JSON行
            data = json.loads(line.strip())
            
            # 确保存在speech字段
            if 'speech' in data:
                # 分割原路径（按##分割，保留后半部分）
                speech_parts = data['speech'].split('##')
                if len(speech_parts) == 2:
                    # 拼接新前缀和原有后半部分
                    data['speech'] = f"{new_prefix}##{speech_parts[1]}"
            
            # 写入修改后的JSON行
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例
if __name__ == "__main__":
    # 输入输出文件路径
    input_jsonl = "/hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/data/star_audiocaps_speecht5/test/caption.jsonl"    # 原始JSONL文件
    output_jsonl = "/hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/data/star_audiocaps_speecht5/test/CLB_env_noise_caption.jsonl"  # 修改后的JSONL文件
    
    # 新的路径前缀（根据需要修改）
    new_path_prefix = "/hpc_stor03/sjtu_home/yixuan.li/work/audio_embeds/cavp_features/speecht5_features/CLB_env_noise_test.h5"
    
    # 执行修改
    modify_jsonl_speech_prefix(input_jsonl, output_jsonl, new_path_prefix)
    print(f"已完成路径修改，结果保存至 {output_jsonl}")
