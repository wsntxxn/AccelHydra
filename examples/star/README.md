# STAR

STAR is the first end-to-end speech-to-audio generation framework. 

## Data preparation

For each generation dataset, the input content information should be organized in a `caption.jsonl`.
Each line in `caption.jsonl` is like:

```JSON
{"audio_id": "a.wav", "speech": "b.h5##a"}
```

The target audio files should be organized in an `audio.jsonl`, with similar formats:

```JSON
{"audio_id": "a.wav", "audio": "b.h5"}
```

We use audiocaps as the dataset and speecht5_vc as speech frontend for example. our pre-processing scripts are in [data_preprocess](./data_preprocess).
You may use them as reference to process your own data.

An example is presented in [data/star_audiocaps_speecht5](./data/star_audiocaps_speecht5)

Then, you **should** add **2** yaml files to configs, the format can be seen in [configs/data/star_audiocaps_speecht5.yaml](./configs/data/star_audiocaps_speecht5.yaml) and [configs/data/datasets/star_audiocaps_speecht5.yaml](./configs/data/datasets/star_audiocaps_speecht5.yaml). 

After that , you **should** add an entry to [configs/data/default.yaml](./configs/data/default.yaml)

## Model Training

To train the model, you could run the following command:

```bash
bash bash_scripts_star/train_star_fm_ft.sh
```

## Inference

To perform inference with trained models, you could use this command:

```bash
bash scripts/infer_star_multi_gpu.sh
```

Ensure your environment is set up and data paths are correct to reproduce results.

## Evaluation

To perform evaluation with inference results, you could use this command:

```bash
bash scripts/eval.sh
```

