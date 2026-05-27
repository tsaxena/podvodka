import argparse
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crowdkit.aggregation import BradleyTerry
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
from trl import RewardTrainer, RewardConfig

sep_token = '</s>'


def sample_pairs(df):
    rows = []

    for desc, desc_df in df.groupby('desc'):
        n = len(desc_df)

        for i in range(n - 1):
            for j in range(i + 1, n):
                prompt_left = desc_df.iloc[i]['prompt']
                prompt_right = desc_df.iloc[j]['prompt']
                if desc_df.iloc[i]['score'] + desc_df.iloc[j]['score'] < 1e-9:
                    y = 0
                else:
                    y = desc_df.iloc[i]['score'] / (desc_df.iloc[i]['score'] + desc_df.iloc[j]['score'])

                left_text = f'{desc}{sep_token}{prompt_left}'
                right_text = f'{desc}{sep_token}{prompt_right}'
                rows.append([left_text, right_text, y])
    return pd.DataFrame(rows, columns=['left', 'right', 'y'])


def sample_aux_pairs(df):
    def sample_neq(n, i):
        j = i
        k = 0
        while i == j and k < 100:
            j = np.random.randint(n)
            k += 1
        return j

    desc_to_prompts = []

    res = []

    words = set()
    for p in df['prompt']:
        for w in p.split():
            words.add(w)
    words = list(words)

    for desc, prompts in df.groupby('image_description'):
        desc_to_prompts.append((desc, prompts['prompt']))

    for i in range(len(desc_to_prompts)):
        prompt_left = desc_to_prompts[i][1].iloc[np.random.randint(len(desc_to_prompts[i][1]))]
        j = sample_neq(len(desc_to_prompts), i)
        prompt_right = desc_to_prompts[j][1].iloc[np.random.randint(len(desc_to_prompts[j][1]))]
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_right}', 1.0])

    for i in range(len(desc_to_prompts)):
        q = np.random.randint(len(desc_to_prompts[i][1]))
        prompt_left = desc_to_prompts[i][1].iloc[q]
        w = sample_neq(len(desc_to_prompts[i][1]), q)
        prompt_left2 = desc_to_prompts[i][1].iloc[w]
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_left2} {prompt_left}', 1.0])

    for i in range(len(desc_to_prompts)):
        prompt_left = desc_to_prompts[i][1].iloc[np.random.randint(len(desc_to_prompts[i][1]))]
        j = sample_neq(len(desc_to_prompts), i)

        n_words = np.random.randint(10)
        prompt_words = []
        for i in range(n_words):
            prompt_words.append(words[np.random.randint(len(words))])
        prompt_right = ' '.join(prompt_words)
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_right}', 1.0])
    return pd.DataFrame(res, columns=['left', 'right', 'y'])


def pairs_to_hf_dataset(df):
    """Convert pairwise dataframe to TRL RewardTrainer format.

    For y >= 0.5, left is chosen; otherwise right is chosen.
    """
    chosen_texts = []
    rejected_texts = []

    for _, row in df.iterrows():
        if row['y'] >= 0.5:
            chosen_texts.append(row['left'])
            rejected_texts.append(row['right'])
        else:
            chosen_texts.append(row['right'])
            rejected_texts.append(row['left'])

    return HFDataset.from_dict({'chosen': chosen_texts, 'rejected': rejected_texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--img_name_map', type=str, default='/mnt/data/img_name_map.csv')
    parser.add_argument('--base_url', type=str, default='https://sdcomparisons.blob.core.windows.net/prompts-comparison/')
    parser.add_argument('--train_data', type=str, default='/mnt/data/train_data.csv')
    parser.add_argument('--output_model', type=str, default='/mnt/data/model')
    parser.add_argument('--wandb_entity', type=str, default='toloka-research')
    parser.add_argument('--wandb_project', type=str, default='prompts_reward_model')
    args = parser.parse_args()

    os.environ['WANDB_ENTITY'] = args.wandb_entity

    res_map = pd.read_csv(args.img_name_map)
    img2prompt = {args.base_url + img_name: prompt for _, img_name, prompt in tqdm(res_map[['image_name', 'prompt']].itertuples(), total=len(res_map))}
    df = pd.read_csv(args.train_data, sep='\t')
    df = df[df['GOLDEN:result'].isna()]
    df['left'] = df['INPUT:left_0'].apply(lambda x: img2prompt[x])
    df['right'] = df['INPUT:right_0'].apply(lambda x: img2prompt[x])
    ann_df = df[['INPUT:prompt', 'left', 'right', 'OUTPUT:result', 'ASSIGNMENT:worker_id']]
    ann_df.columns = ['desc', 'left', 'right', 'label', 'worker']
    ann_df['label'] = ann_df.apply(lambda row: row['left'] if row['label'] == 'left' else row['right'], axis=1)

    scores = {}

    for descr, d in ann_df.groupby('desc'):
        scores[descr] = BradleyTerry(100).fit_predict(d)

    train_desc, val_desc = train_test_split(list(scores.keys()), test_size=0.2, random_state=42)

    ds_train = []
    ds_val = []

    for desc, prompt_scores in scores.items():
        for prompt, score in prompt_scores.items():
            line = [desc, prompt, score]
            if desc in train_desc:
                ds_train.append(line)
            else:
                ds_val.append(line)
    ds_train = pd.DataFrame(ds_train, columns=['desc', 'prompt', 'score'])
    ds_val = pd.DataFrame(ds_val, columns=['desc', 'prompt', 'score'])

    ds_train = sample_pairs(ds_train)
    ds_val = sample_pairs(ds_val)

    res_map_train = res_map[~res_map['image_description'].isin(set(val_desc))]
    ds_aux = sample_aux_pairs(res_map_train)

    # Merge aux pairs into training data (aux pairs always have y=1.0 so left is chosen)
    ds_train_combined = pd.concat([ds_train, ds_aux], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1)

    train_dataset = pairs_to_hf_dataset(ds_train_combined)
    val_dataset = pairs_to_hf_dataset(ds_val)

    training_args = RewardConfig(
        output_dir=args.output_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=500,
        eval_strategy='epoch',
        save_strategy='epoch',
        report_to='wandb',
        run_name=args.wandb_project,
        max_length=512,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_model)


if __name__ == '__main__':
    main()
