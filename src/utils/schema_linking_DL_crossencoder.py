import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RUN_MODE = 'train'
VERBOSE = True
SAVE_DIR = ''
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class GlobalTablePool:

    def __init__(self):
        self.table_schemas = {}
        self.project_tables = defaultdict(set)  
        self.project_folder_mapping = {}  

    def add_table(self, project_name: str, table_name: str, columns: dict):
        self.table_schemas[table_name] = columns
        self.project_tables[project_name].add(table_name)

    def get_all_tables(self) -> List[str]:
        return list(self.table_schemas.keys())

    def get_project_tables(self, project_name: str) -> List[str]:
        return list(self.project_tables.get(project_name, []))

    def get_other_project_tables(self, project_name: str, sample_size: int = 50) -> List[str]:
        all_tables = set(self.get_all_tables())
        current_project_tables = set(self.get_project_tables(project_name))
        other_tables = list(all_tables - current_project_tables)

        if len(other_tables) <= sample_size:
            return other_tables
        return random.sample(other_tables, sample_size)

    def get_table_schema(self, table_name: str) -> dict:
        return self.table_schemas.get(table_name, {})

    def get_project_folder_name(self, project_name: str) -> str:
        normalized = project_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('：', '').replace(':', '')
        return self.project_folder_mapping.get(normalized, None)

    def build_from_csv_folders(self, dataset_root: str):

        total_projects = 0
        total_tables = 0

        for project_folder in os.listdir(dataset_root):
            project_path = os.path.join(dataset_root, project_folder)

            if not os.path.isdir(project_path):
                continue
            if project_folder.endswith('.json') or project_folder.endswith('.txt'):
                continue

            project_base_name = project_folder

            project_display_name = self._infer_project_display_name_from_folder(project_folder, project_path)
            if project_display_name:
                normalized_name = project_display_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('：', '').replace(':', '')
                self.project_folder_mapping[normalized_name] = project_folder

            for csv_file in os.listdir(project_path):
                if not csv_file.endswith('.csv'):
                    continue

                csv_path = os.path.join(project_path, csv_file)
                table_name = csv_file.replace('.csv', '')

                if table_name.startswith('node_'):
                    project_display_name = self._infer_project_display_name_from_folder(project_folder, project_path)
                    if project_display_name:
                        table_name = f"{project_display_name}_{table_name}"

                try:
                    df = pd.read_csv(csv_path, nrows=0)
                    columns = {col: 0 for col in df.columns}
                    self.add_table(project_base_name, table_name, columns)
                    total_tables += 1
                except Exception as e:
                    pass

            total_projects += 1

        return self

    def build_from_json(self, json_file_path: str, add_project_prefix_to_nodes: bool = False):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for project_name, queries in data.items():
            project_display_name = self._infer_project_display_name(queries)

            for query_item in queries:
                table_relevance = query_item.get('table_relevance', {})
                column_relevance = query_item.get('column_relevance', {})

                for table_name in table_relevance.keys():
                    columns = column_relevance.get(table_name, {})

                    if add_project_prefix_to_nodes and table_name.startswith('node_'):
                        if project_display_name:
                            prefixed_table_name = f"{project_display_name}_{table_name}"
                        else:
                            prefixed_table_name = table_name
                        self.add_table(project_name, prefixed_table_name, columns)
                    else:
                        self.add_table(project_name, table_name, columns)
        return self

    def _infer_project_display_name(self, queries):
        if not queries:
            return None

        all_tables = []
        for query in queries:
            table_relevance = query.get('table_relevance', {})
            for table_name in table_relevance.keys():
                if not table_name.startswith('node_'):
                    all_tables.append(table_name)

        if not all_tables:
            return None

        common_prefix = all_tables[0]
        for table in all_tables[1:]:
            i = 0
            while i < len(common_prefix) and i < len(table) and common_prefix[i] == table[i]:
                i += 1
            common_prefix = common_prefix[:i]

        common_prefix = common_prefix.rstrip('_')
        last_underscore = common_prefix.rfind('_')
        if last_underscore > 0:
            common_prefix = common_prefix[:last_underscore]

        return common_prefix

    def _infer_project_display_name_from_folder(self, project_folder: str, project_path: str = None):
        suffixes_to_remove = [
            '_normal', '_no_overlap', '_no_join_cols',
            '_with_overlap', '_complex', '_simple'
        ]

        project_display_name = project_folder

        for suffix in suffixes_to_remove:
            if project_display_name.endswith(suffix):
                project_display_name = project_display_name[:-len(suffix)]
                break

        return project_display_name

class CrossEncoderDataset(Dataset):

    def __init__(self, data_list, tokenizer, global_table_pool: GlobalTablePool,
                 max_length=512, neg_sample_size=7, is_training=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.global_table_pool = global_table_pool
        self.neg_sample_size = neg_sample_size
        self.is_training = is_training
        self.examples = []
        HARD_NEG_RATIO = 2
        EASY_NEG_RATIO = 5 
        
        if neg_sample_size != (HARD_NEG_RATIO + EASY_NEG_RATIO):
             total = HARD_NEG_RATIO + EASY_NEG_RATIO
             HARD_NEG_RATIO = int(neg_sample_size * (HARD_NEG_RATIO/total))
             EASY_NEG_RATIO = neg_sample_size - HARD_NEG_RATIO

        for item in tqdm(data_list, desc="Crossencoder_Dataset", disable=not VERBOSE):
            query = item.get("query") or item.get("question", "")
            if not query:
                continue
            
            project_name = item.get("project_name", "unknown")
            table_relevance = item.get("table_relevance", {})
            column_relevance = item.get("column_relevance", {})

            pos_tables = []
            for table_name, relevance in table_relevance.items():
                if relevance == 1:
                    columns = column_relevance.get(table_name, {})
                    self.examples.append({
                        'query': query,
                        'table_name': table_name,
                        'columns': columns,
                        'label': 1
                    })
                    pos_tables.append(table_name)
            
            num_pos = len(pos_tables)
            if num_pos == 0: 
                continue

            project_tables = set(self.global_table_pool.get_project_tables(project_name))
            irrelevant_tables_in_project = [t for t in project_tables if table_relevance.get(t, 0) == 0]
            
            other_project_tables = self.global_table_pool.get_other_project_tables(
                project_name, sample_size=(EASY_NEG_RATIO * num_pos) + 20
            )

            
            target_hard_count = num_pos * HARD_NEG_RATIO
            target_easy_count = num_pos * EASY_NEG_RATIO
            
            current_hard_negs = []
            if len(irrelevant_tables_in_project) >= target_hard_count:
                current_hard_negs = random.sample(irrelevant_tables_in_project, target_hard_count)
            else:
                current_hard_negs = irrelevant_tables_in_project
                target_easy_count += (target_hard_count - len(current_hard_negs))
            
            current_easy_negs = []
            if len(other_project_tables) >= target_easy_count:
                current_easy_negs = random.sample(other_project_tables, target_easy_count)
            else:
                current_easy_negs = other_project_tables

            for neg_table in current_hard_negs + current_easy_negs:
                columns = self.global_table_pool.get_table_schema(neg_table)
                self.examples.append({
                    'query': query,
                    'table_name': neg_table,
                    'columns': columns,
                    'label': 0
                })

        pos_count = sum(1 for ex in self.examples if ex['label'] == 1)
        neg_count = len(self.examples) - pos_count
        ratio = neg_count/pos_count if pos_count > 0 else 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        columns_dict = example['columns']
        col_names = list(columns_dict.keys())
        if self.is_training:
             random.shuffle(col_names)
        
        query_text = example['query']
        table_text = self._build_table_text(example['table_name'], col_names)
        
        combined_text = f"{query_text} {self.tokenizer.sep_token} {table_text}"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(example['label'], dtype=torch.float)
        }

    def _build_table_text(self, table_name, columns):
        if isinstance(columns, dict):
            cols = list(columns.keys())
        else:
            cols = columns
            
        parts = [table_name.replace('_', ' ')]
        for col in cols[:30]:
            parts.append(col.replace('_', ' '))
        return ' | '.join(parts)

class CrossEncoderSchemaLinker(nn.Module):

    def __init__(self, model_name='microsoft/deberta-v3-large', dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1) 
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)  # [batch_size, 1]
        return logits.squeeze(-1)  # [batch_size]


class CrossEncoderTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, val_data,
                 train_table_pool, test_data=None, test_table_pool=None,
                 device='cuda', lr=2e-5, accumulation_steps=1):
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_data = val_data
        self.test_data = test_data
        self.train_table_pool = train_table_pool
        self.test_table_pool = test_table_pool
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        pos_weight = torch.tensor([6.0]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_dataloader, desc="Training")
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(logits, labels)
            
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (step + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            total_loss += loss.item() * self.accumulation_steps
            acc = correct / labels.size(0)
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps, 'acc': acc})

        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_dataloader)
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def evaluate_on_full_table_pool(self, data_list, top_k=20, max_samples=None,
                                    use_global_pool=True, save_detailed=True):
        self.model.eval()
        results = []

        original_size = len(data_list)
        if max_samples is not None and max_samples < len(data_list):
            data_list = data_list[:max_samples]

        if use_global_pool:
            table_pool = self.test_table_pool
            pool_size = len(table_pool.get_all_tables())
        else:
            table_pool = self.train_table_pool

        with torch.no_grad():
            for idx, item in enumerate(tqdm(data_list, desc="Evaluating")):
                query = item['query']
                project_name = item.get('project_name', 'unknown')
                true_tables = set(item.get('query_related_table', []))
                table_relevance = item.get('table_relevance', {})
                column_relevance = item.get('column_relevance', {})

                if use_global_pool:
                    all_tables = table_pool.get_all_tables()
                else:
                    all_tables = list(table_relevance.keys())

                scores = self._compute_scores_batch(query, all_tables, table_pool, batch_size=32)
                table_scores = dict(zip(all_tables, scores))

                sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                num_to_select = min(top_k, len(sorted_tables))
                predicted_tables = set([t for t, s in sorted_tables[:num_to_select]])

                tp = len(predicted_tables & true_tables)
                fp = len(predicted_tables - true_tables)
                fn = len(true_tables - predicted_tables)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                top_20_scores = sorted_tables[:20]
                gt_scores = {t: table_scores[t] for t in true_tables if t in table_scores}

                if idx < 3: 
                    all_scores = list(table_scores.values())
                    gt_score_list = list(gt_scores.values())
                    non_gt_scores = [s for t, s in table_scores.items() if t not in true_tables]
             
                    for t in list(true_tables)[:3]:
                        if t in table_scores:
                            rank = sorted(table_scores.items(), key=lambda x: x[1], reverse=True).index((t, table_scores[t])) + 1
                            print(f"   {t[:60]}: {table_scores[t]:.6f} (排名 {rank})")
                    
                    if gt_score_list:
                        gt_min = min(gt_score_list)
                        gt_max = max(gt_score_list)
                        non_gt_max = max(non_gt_scores)
                        overlap_count = sum(1 for s in non_gt_scores if s > gt_min)

                detailed_scores = []
                seen_tables = set()

                for table, score in top_20_scores:
                    detailed_scores.append([table, float(score)])
                    seen_tables.add(table)

                for table in sorted(gt_scores.keys(), key=lambda t: gt_scores[t], reverse=True):
                    if table not in seen_tables:
                        detailed_scores.append([table, float(gt_scores[table])])

                result = {
                    'sample_id': idx,
                    'query': query,
                    'table_scores': detailed_scores,
                    'threshold': None,
                    'top_k': top_k,
                    'selection_method': 'top_k',
                    'query_related_table_tables': list(true_tables),
                    'predicted_tables': list(predicted_tables),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }

                results.append(result)

        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])

        return results, {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}

    def _compute_scores_batch(self, query, table_names, table_pool, batch_size=32):
        tokenizer = self.model.module.tokenizer if self.multi_gpu else self.model.tokenizer
        
        scores = []
        num_tables = len(table_names)

        for i in range(0, num_tables, batch_size):
            batch_tables = table_names[i:i+batch_size]
            
            combined_texts = []
            for table_name in batch_tables:
                columns = table_pool.get_table_schema(table_name)
                table_text = self._build_table_text(table_name, columns)
                combined_text = f"{query} {tokenizer.sep_token} {table_text}"
                combined_texts.append(combined_text)
            
            encodings = tokenizer(
                combined_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            logits = self.model(encodings['input_ids'], encodings['attention_mask'])
            
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            scores.extend(batch_scores)

        return scores

    def _build_table_text(self, table_name, columns):
        parts = [table_name.replace('_', ' ')]
        for col in list(columns.keys())[:30]:
            parts.append(col.replace('_', ' '))
        return ' | '.join(parts)

    def train(self, num_epochs=15):
        best_f1 = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(f"\n Epoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if True: 
                model_to_save = self.model.module if self.multi_gpu else self.model
                checkpoint_path = os.path.join(SAVE_DIR, f'model_epoch{epoch+1}.pt')
                torch.save(model_to_save.state_dict(), checkpoint_path)

                if self.val_data:
                    val_results, val_metrics = self.evaluate_on_full_table_pool(
                        self.val_data, top_k=20, max_samples=10, use_global_pool=False, save_detailed=False
                    )

                if self.test_data:
                    test_results, test_metrics = self.evaluate_on_full_table_pool(
                        self.test_data, top_k=20, max_samples=10, use_global_pool=True, save_detailed=True
                    )

                    if test_metrics['f1'] > best_f1:
                        best_f1 = test_metrics['f1']
                        best_model_path = os.path.join(SAVE_DIR, 'best_model.pt')
                        torch.save(model_to_save.state_dict(), best_model_path)

                    test_results_file = os.path.join(SAVE_DIR, f'test_results_epoch{epoch+1}.json')
                    with open(test_results_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'epoch': epoch + 1,
                            'metrics': test_metrics,
                            'detailed_results': test_results
                        }, f, ensure_ascii=False, indent=2)

        if self.test_data:
            test_results, test_metrics = self.evaluate_on_full_table_pool(
                self.test_data, top_k=20, max_samples=None, use_global_pool=True, save_detailed=True
            )

            results_file = os.path.join(SAVE_DIR, 'test_results_final.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': test_metrics,
                    'detailed_results': test_results
                }, f, ensure_ascii=False, indent=2)


def load_data_with_global_pool(train_file, test_file, use_all_train_data=True, train_ratio=0.9,
                               csv_dataset_root=None):
    train_table_pool = GlobalTablePool()
    train_table_pool.build_from_json(train_file, add_project_prefix_to_nodes=False)

    test_table_pool = GlobalTablePool()
    if csv_dataset_root and os.path.exists(csv_dataset_root):
        test_table_pool.build_from_csv_folders(csv_dataset_root)
    else:
        test_table_pool.build_from_json(test_file, add_project_prefix_to_nodes=True)

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data_dict = json.load(f)

    all_train_examples = []
    for project_name, queries in train_data_dict.items():
        for item in queries:
            item['project_name'] = project_name
            all_train_examples.append(item)

    if use_all_train_data:
        sampled_train = all_train_examples
    else:
        sample_size = int(len(all_train_examples) * 0.5)
        sampled_train = random.sample(all_train_examples, sample_size)

    train_size = int(len(sampled_train) * train_ratio)
    train_data = sampled_train[:train_size]
    val_data = sampled_train[train_size:]

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data_dict = json.load(f)

    test_data = []
    for project_name, queries in test_data_dict.items():
        project_folder = test_table_pool.get_project_folder_name(project_name)
        if project_folder:
            project_display_name = _extract_display_name_from_folder(project_folder)
        else:
            project_display_name = _infer_project_display_name_from_queries(queries)

        for item in queries:
            item['project_name'] = project_name

            if 'query_related_table' in item:
                renamed_tables = []
                for table in item['query_related_table']:
                    if table.startswith('node_') and project_display_name:
                        renamed_tables.append(f"{project_display_name}_{table}")
                    else:
                        renamed_tables.append(table)
                item['query_related_table'] = renamed_tables

            if 'table_relevance' in item:
                renamed_table_relevance = {}
                for table, relevance in item['table_relevance'].items():
                    if table.startswith('node_') and project_display_name:
                        renamed_table_relevance[f"{project_display_name}_{table}"] = relevance
                    else:
                        renamed_table_relevance[table] = relevance
                item['table_relevance'] = renamed_table_relevance

            if 'column_relevance' in item:
                renamed_column_relevance = {}
                for table, columns in item['column_relevance'].items():
                    if table.startswith('node_') and project_display_name:
                        renamed_column_relevance[f"{project_display_name}_{table}"] = columns
                    else:
                        renamed_column_relevance[table] = columns
                item['column_relevance'] = renamed_column_relevance

            test_data.append(item)


    return train_data, val_data, test_data, train_table_pool, test_table_pool


def _extract_display_name_from_folder(folder_name: str) -> str:
    suffixes_to_remove = ['_normal', '_no_overlap', '_no_join_cols',
                        '_with_overlap', '_complex', '_simple']

    display_name = folder_name
    for suffix in suffixes_to_remove:
        if display_name.endswith(suffix):
            display_name = display_name[:-len(suffix)]
            break

    return display_name


def _infer_project_display_name_from_queries(queries):
    if not queries:
        return None

    all_tables = []
    for query in queries:
        table_relevance = query.get('table_relevance', {})
        for table_name in table_relevance.keys():
            if not table_name.startswith('node_'):
                all_tables.append(table_name)

    if not all_tables:
        return None

    common_prefix = all_tables[0]
    for table in all_tables[1:]:
        i = 0
        while i < len(common_prefix) and i < len(table) and common_prefix[i] == table[i]:
            i += 1
        common_prefix = common_prefix[:i]

    common_prefix = common_prefix.rstrip('_')
    last_underscore = common_prefix.rfind('_')
    if last_underscore > 0:
        common_prefix = common_prefix[:last_underscore]

    return common_prefix if common_prefix else None


def main():
    train_file = ''
    test_file = ''
    csv_dataset_root = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = ''

    train_data, val_data, test_data, train_table_pool, test_table_pool = load_data_with_global_pool(
        train_file, test_file, use_all_train_data=True, train_ratio=0.9,
        csv_dataset_root=csv_dataset_root
    )
    model = CrossEncoderSchemaLinker(model_name=model_path, dropout=0.1)
    tokenizer = model.tokenizer

    train_dataset = CrossEncoderDataset(
        train_data, tokenizer, train_table_pool,
        max_length=512, neg_sample_size=7, is_training=True
    )
    val_dataset = CrossEncoderDataset(
        val_data, tokenizer, train_table_pool,
        max_length=512, neg_sample_size=7, is_training=False
    )

    batch_size = 4

    num_workers = 4

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    trainer = CrossEncoderTrainer(
        model, train_dataloader, val_dataloader, val_data,
        train_table_pool, test_data=test_data, test_table_pool=test_table_pool,
        device=device, lr=5e-6, accumulation_steps=8 
    )

    trainer.train(num_epochs=15)

if __name__ == '__main__':
    main()