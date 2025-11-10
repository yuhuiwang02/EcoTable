import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torch.optim import AdamW
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

RUN_MODE = 'train'  
TEST_FULL_DATASET = True  

TEST_MODEL_PATH = 'path/to/test_model'

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

SAVE_DIR = 'path/to/dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

VERBOSE = False 
def validate_data_consistency(data_list):
    inconsistent_samples = []
    for idx, item in enumerate(data_list):
        query = item.get('query', '')
        query_related_table = item.get('query_related_table', [])
        table_relevance = item.get('table_relevance', {})

        for table in query_related_table:
            if table not in table_relevance:
                inconsistent_samples.append({
                    'sample_idx': idx,
                    'issue': f'Ground truth table "{table}" not in table_relevance'
                })
            elif table_relevance[table] != 1:
                inconsistent_samples.append({
                    'sample_idx': idx,
                    'issue': f'Ground truth table "{table}" marked as irrelevant'
                })

        relevant_tables = [table for table, label in table_relevance.items() if label == 1]
        for table in relevant_tables:
            if table not in query_related_table:
                inconsistent_samples.append({
                    'sample_idx': idx,
                    'issue': f'Table "{table}" marked as relevant but not in query_related_table'
                })
    return inconsistent_samples

class SinglePairDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512,
                 pos_neg_balance=True, max_pairs_per_item=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        validate_data_consistency(data_list)

        for item in tqdm(data_list, desc="📊 构建数据集", disable=not VERBOSE):
            query = item["query"]
            table_relevance = item.get("table_relevance", {})
            column_relevance = item.get("column_relevance", {})
            if not table_relevance:
                continue

            pairs = []
            for table, label in table_relevance.items():
                seq = self._build_table_sequence(query, table, column_relevance.get(table, {}))
                pairs.append((query, seq, int(label)))

            if pos_neg_balance and len(pairs) > 0:
                pos_pairs = [p for p in pairs if p[2] == 1]
                neg_pairs = [p for p in pairs if p[2] == 0]
                if len(pos_pairs) > 0 and len(neg_pairs) > 0:
                    k = min(len(pos_pairs), len(neg_pairs))
                    pairs = random.sample(pos_pairs, k) + random.sample(neg_pairs, k)

            if max_pairs_per_item is not None and len(pairs) > max_pairs_per_item:
                pairs = random.sample(pairs, max_pairs_per_item)

            for q, seq, y in pairs:
                self.examples.append({
                    'query': q,
                    'candidate_sequence': seq,
                    'label': y
                })

    def _build_table_sequence(self, query, table_name, columns):
        parts = [query, '[TAB]', table_name.replace('_', ' ')]
        for col in columns.keys():
            parts.extend(['[COL]', col.replace('_', ' ')])
        return ' '.join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        q_enc = self.tokenizer(
            ex['query'], truncation=True, padding='max_length',
            max_length=self.max_length // 2, return_tensors='pt'
        )
        c_enc = self.tokenizer(
            ex['candidate_sequence'], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'query_input_ids': q_enc['input_ids'].squeeze(),
            'query_attention_mask': q_enc['attention_mask'].squeeze(),
            'cand_input_ids': c_enc['input_ids'].squeeze(),
            'cand_attention_mask': c_enc['attention_mask'].squeeze(),
            'label': torch.tensor(ex['label'], dtype=torch.float)
        }

class ContrastiveSchemaLinker(nn.Module):
    def __init__(self, model_path, embedding_dim=256):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.encoder = RobertaModel.from_pretrained(model_path)

        special_tokens = ['[TAB]', '[COL]']
        num_added_tokens = self.tokenizer.add_tokens(special_tokens)
        if num_added_tokens > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def encode_sequence(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        emb = self.projection(cls)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, query_ids, query_mask, cand_ids, cand_mask):
        q_emb = self.encode_sequence(query_ids, query_mask)
        c_emb = self.encode_sequence(cand_ids, cand_mask)
        return {'query_embedding': q_emb, 'cand_embedding': c_emb}

    @torch.no_grad()
    def pair_score(self, query_ids, query_mask, cand_ids, cand_mask):
        q = self.encode_sequence(query_ids, query_mask)
        c = self.encode_sequence(cand_ids, cand_mask)
        score = torch.cosine_similarity(q, c, dim=1)
        return score

class SinglePairLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_weight=None):
        super().__init__()
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, query_emb, cand_emb, labels):
        cos_sim = torch.cosine_similarity(query_emb, cand_emb, dim=1)
        logits = cos_sim / self.temperature
        loss = self.bce(logits, labels)
        return loss, logits

class ThresholdOptimizer:
    def __init__(self, model, val_data, device, temperature=0.07):
        self.model = model
        self.val_data = val_data
        self.device = device
        self.temperature = temperature
        self.best_threshold = 0.5
        self.best_f1 = 0.0

    def find_optimal_threshold(self):
        all_scores, all_labels = [], []

        for item in tqdm(self.val_data, desc="🎯 优化阈值", leave=False):
            query = item['query']
            table_relevance = item['table_relevance']
            column_relevance = item['column_relevance']

            q_enc = self.model.tokenizer(query, truncation=True, padding='max_length',
                                         max_length=256, return_tensors='pt')
            q_ids = q_enc['input_ids'].to(self.device)
            q_mask = q_enc['attention_mask'].to(self.device)
            with torch.no_grad():
                q_emb = self.model.encode_sequence(q_ids, q_mask)

            for table, label in table_relevance.items():
                parts = [query, '[TAB]', table.replace('_', ' ')]
                for col in column_relevance.get(table, {}).keys():
                    parts.extend(['[COL]', col.replace('_', ' ')])
                cand = ' '.join(parts)

                c_enc = self.model.tokenizer(cand, truncation=True, padding='max_length',
                                             max_length=512, return_tensors='pt')
                c_ids = c_enc['input_ids'].to(self.device)
                c_mask = c_enc['attention_mask'].to(self.device)
                with torch.no_grad():
                    c_emb = self.model.encode_sequence(c_ids, c_mask)
                    cos = torch.cosine_similarity(q_emb, c_emb, dim=1).item()

                prob = 1 / (1 + np.exp(-(cos / self.temperature)))
                all_scores.append(prob)
                all_labels.append(label)

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            preds = [1 if s >= t else 0 for s in all_scores]
            p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.best_threshold = best_t
        self.best_f1 = best_f1
        return best_t

class ImprovedSinglePairTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, val_data_raw, test_data=None,
                 device='cuda', temperature=0.07, manual_threshold=None, lr=2e-5):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_data_raw = val_data_raw
        self.test_data = test_data
        self.device = device
        self.temperature = temperature

        self.criterion = SinglePairLoss(temperature=self.temperature)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, len(train_dataloader) * 10)
        )

        self.history = defaultdict(list)
        self.validation_results = []

        if manual_threshold is not None:
            self.manual_threshold = manual_threshold
            self.optimal_threshold = manual_threshold
        else:
            self.manual_threshold = None
            self.optimal_threshold = 0.5

    def set_threshold(self, threshold):
        self.manual_threshold = threshold
        self.optimal_threshold = threshold

    def train_epoch(self, epoch_num, total_epochs):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        running_acc = 0.0

        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"🚀 Epoch {epoch_num}/{total_epochs}",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch in progress_bar:
            qi = batch['query_input_ids'].to(self.device)
            qm = batch['query_attention_mask'].to(self.device)
            ci = batch['cand_input_ids'].to(self.device)
            cm = batch['cand_attention_mask'].to(self.device)
            y = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(qi, qm, ci, cm)
            loss, logits = self.criterion(outputs['query_embedding'], outputs['cand_embedding'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                acc = (preds == y).float().mean().item()

            total_loss += loss.item()
            running_acc += acc
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.3f}'
            })

        avg_loss = total_loss / max(1, num_batches)
        train_accuracy = running_acc / max(1, num_batches)
        return avg_loss, train_accuracy

    def evaluate(self, dataloader=None, threshold=None):
        if dataloader is None:
            dataloader = self.val_dataloader
        if threshold is None:
            threshold = self.optimal_threshold

        self.model.eval()
        total_loss, num_batches = 0.0, 0
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="📈 验证中", leave=False, ncols=80):
                qi = batch['query_input_ids'].to(self.device)
                qm = batch['query_attention_mask'].to(self.device)
                ci = batch['cand_input_ids'].to(self.device)
                cm = batch['cand_attention_mask'].to(self.device)
                y = batch['label'].to(self.device)

                outputs = self.model(qi, qm, ci, cm)
                loss, logits = self.criterion(outputs['query_embedding'], outputs['cand_embedding'], y)
                total_loss += loss.item()
                num_batches += 1

                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                y_prob.extend(probs)
                y_true.extend(y.detach().cpu().numpy().tolist())

        preds = [1 if p >= threshold else 0 for p in y_prob]
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        acc = accuracy_score(y_true, preds)
        return {
            'validation_loss': total_loss / max(1, num_batches),
            'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc
        }

    def optimize_threshold(self):
        if self.manual_threshold is not None:
            return self.manual_threshold
        threshold_optimizer = ThresholdOptimizer(self.model, self.val_data_raw, self.device, temperature=self.temperature)
        self.optimal_threshold = threshold_optimizer.find_optimal_threshold()
        return self.optimal_threshold

    def test_multiple_thresholds(self, test_data, thresholds_to_test=None):
        if thresholds_to_test is None:
            thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold_results = []
        for threshold in tqdm(thresholds_to_test, desc="🔍 测试阈值"):
            original_threshold = self.optimal_threshold
            self.optimal_threshold = threshold

            results, _ = self.inference_with_threshold(
                test_data, use_learned_threshold=True, save_detailed_results=False
            )
            avg_precision = np.mean([r['precision'] for r in results]) if results else 0.0
            avg_recall = np.mean([r['recall'] for r in results]) if results else 0.0
            avg_f1 = np.mean([r['f1'] for r in results]) if results else 0.0

            threshold_results.append({
                'threshold': threshold,
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1)
            })

            self.optimal_threshold = original_threshold

        best_result = max(threshold_results, key=lambda x: x['f1']) if threshold_results else {'threshold': 0.5, 'f1': 0.0}
        
        threshold_test_file = os.path.join(SAVE_DIR, 'threshold_comparison_results_singlepair.json')
        with open(threshold_test_file, 'w', encoding='utf-8') as f:
            json.dump(threshold_results, f, ensure_ascii=False, indent=2)
        self.plot_threshold_comparison(threshold_results)
        
        return threshold_results, best_result

    def plot_threshold_comparison(self, threshold_results):
        thresholds = [r['threshold'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1s = [r['f1'] for r in threshold_results]

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2, markersize=6)
        plt.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2, markersize=6)
        plt.plot(thresholds, f1s, 'g-^', label='F1 Score', linewidth=2, markersize=6)

        best_idx = np.argmax(f1s) if f1s else 0
        if f1s:
            plt.axvline(x=thresholds[best_idx], color='gray', linestyle='--', alpha=0.7)
            plt.text(thresholds[best_idx], f1s[best_idx] + 0.02,
                     f'Best F1\n({thresholds[best_idx]:.1f}, {f1s[best_idx]:.3f})',
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.xlabel('阈值(概率域)', fontsize=12)
        plt.ylabel('性能指标', fontsize=12)
        plt.title('不同阈值下的性能表现', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        if thresholds:
            plt.xlim(min(thresholds) - 0.05, max(thresholds) + 0.05)
        plt.ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'threshold_comparison_singlepair.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def inference_with_threshold(self, test_data, use_learned_threshold=True,
                                 custom_threshold=None, save_detailed_results=True,
                                 use_top_k_percent=False, top_k_percent=0.6,
                                 return_full_data=False, original_data=None):
        self.model.eval()
        results = []
        detailed_results = []
        full_data_with_predictions = []

        if custom_threshold is not None:
            threshold = custom_threshold
        elif use_learned_threshold:
            threshold = self.optimal_threshold
        else:
            threshold = 0.5

        # 创建索引映射（如果提供了original_data）
        if original_data:
            query_to_original = {item['query']: item for item in original_data}
        else:
            query_to_original = {}

        for idx, item in enumerate(tqdm(test_data, desc="🔮 推理中", leave=False)):
            query = item['query']
            query_related_table = item.get('query_related_table', [])
            table_relevance = item.get('table_relevance', {})
            column_relevance = item.get('column_relevance', {})

            all_tables = list(table_relevance.keys())
            if not all_tables:
                continue

            q_enc = self.model.tokenizer(query, truncation=True, padding='max_length',
                                         max_length=256, return_tensors='pt')
            qi = q_enc['input_ids'].to(self.device)
            qm = q_enc['attention_mask'].to(self.device)
            with torch.no_grad():
                q_emb = self.model.encode_sequence(qi, qm)

            predictions = {}
            table_scores = []

            # 计算所有表的得分
            with torch.no_grad():
                for table in all_tables:
                    parts = [query, '[TAB]', table.replace('_', ' ')]
                    for col in column_relevance.get(table, {}).keys():
                        parts.extend(['[COL]', col.replace('_', ' ')])
                    cand = ' '.join(parts)

                    c_enc = self.model.tokenizer(cand, truncation=True, padding='max_length',
                                                 max_length=512, return_tensors='pt')
                    ci = c_enc['input_ids'].to(self.device)
                    cm = c_enc['attention_mask'].to(self.device)

                    c_emb = self.model.encode_sequence(ci, cm)
                    cos = torch.cosine_similarity(q_emb, c_emb, dim=1).item()
                    prob = 1 / (1 + np.exp(-(cos / self.temperature)))
                    table_scores.append((table, float(prob)))

            # 根据筛选方式决定预测结果
            if use_top_k_percent:
                table_scores.sort(key=lambda x: x[1], reverse=True)
                k = max(1, int(len(table_scores) * top_k_percent))
                top_k_tables = set([t for t, _ in table_scores[:k]])
                for table in all_tables:
                    predictions[table] = 1 if table in top_k_tables else 0
            else:
                for table, prob in table_scores:
                    predictions[table] = 1 if prob >= threshold else 0
                table_scores.sort(key=lambda x: x[1], reverse=True)

            y_true = [table_relevance.get(t, 0) for t in all_tables]
            y_pred = [predictions.get(t, 0) for t in all_tables]
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            result = {
                'query': query,
                'predictions': predictions,
                'query_related_table': query_related_table,
                'labels': table_relevance,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'threshold_used': float(threshold) if not use_top_k_percent else None,
                'top_k_percent_used': float(top_k_percent) if use_top_k_percent else None,
                'selection_method': 'top_k_percent' if use_top_k_percent else 'threshold'
            }
            results.append(result)

            if save_detailed_results:
                predicted_tables = [t for t, pred in predictions.items() if pred == 1]
                detailed_result = {
                    'sample_id': idx,
                    'query': query,
                    'all_tables': all_tables,
                    'table_scores': table_scores,
                    'threshold': float(threshold) if not use_top_k_percent else None,
                    'top_k_percent': float(top_k_percent) if use_top_k_percent else None,
                    'selection_method': 'top_k_percent' if use_top_k_percent else 'threshold',
                    'query_related_table_tables': query_related_table,
                    'predicted_tables': predicted_tables,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }
                detailed_results.append(detailed_result)

            if return_full_data:
                if query in query_to_original:
                    original_item = query_to_original[query].copy()
                else:
                    original_item = item.copy()
                
                predicted_tables = [t for t, pred in predictions.items() if pred == 1]
                original_item['predicted_result'] = predicted_tables
                
                full_data_with_predictions.append(original_item)

        if save_detailed_results:
            if use_top_k_percent:
                method_str = f"top{int(top_k_percent*100)}"
            else:
                threshold_str = f"{threshold:.3f}".replace('.', '_')
                method_str = f"threshold_{threshold_str}"
            detailed_results_file = os.path.join(SAVE_DIR, f'detailed_results_{method_str}_singlepair.json')
            with open(detailed_results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        if return_full_data:
            return results, detailed_results if save_detailed_results else results, full_data_with_predictions
        else:
            return results, detailed_results if save_detailed_results else results

    def _save_validation_results(self):
        validation_file = os.path.join(SAVE_DIR, 'singlepair_validation_results_final.json')
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)

    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = self.history['epoch']

        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [result['learning_rate'] for result in self.validation_results] if self.validation_results else []
            if lrs:
                axes[1, 0].plot(epochs, lrs, 'g-', label='Learning Rate')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].text(0.1, 0.8, f'Temperature: {self.temperature}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Optimal Threshold: {self.optimal_threshold:.4f}',
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Info')
        axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'singlepair_training_curves_final.png'), dpi=300)
        plt.close()

    def train(self, epochs):
        best_f1 = 0.0
        print(f"\n{'='*60}")
        print(f"🚀 开始训练 - 共 {epochs} 个 Epoch")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(epoch + 1, epochs)
            
            val_metrics = self.evaluate(self.val_dataloader, threshold=self.optimal_threshold)
            
            print(f"\n📊 Epoch {epoch + 1}/{epochs} 结果:")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")
            print(f"   Valid - Loss: {val_metrics['validation_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"   Valid - P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

            validation_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'validation_metrics': val_metrics,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else 2e-5
            }
            self.validation_results.append(validation_result)

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_metrics['validation_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'f1': best_f1,
                    'epoch': epoch + 1,
                    'temperature': self.temperature,
                    'manual_threshold': self.manual_threshold,
                    'opt_threshold': self.optimal_threshold
                }, os.path.join(SAVE_DIR, 'best_singlepair_model_final.pth'))

            self._save_validation_results()
            self.scheduler.step()


        if self.manual_threshold is None:
            self.optimize_threshold()

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimal_threshold': self.optimal_threshold,
            'manual_threshold': self.manual_threshold,
            'temperature': self.temperature
        }, os.path.join(SAVE_DIR, 'final_singlepair_model_with_threshold.pth'))

        self.plot_training_curves()

def load_and_split_data(
    data_file_path, 
    sampling_strategy='default',
    sample_ratio=None,
    specific_projects=None,
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15, 
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_examples = []
    project_examples = defaultdict(list)
    
    for project_name, project_data in data.items():
        for item in project_data:
            item['project_name'] = project_name
            if 'query' not in item:
                continue
            item.setdefault('query_related_table', [])
            item.setdefault('table_relevance', {})
            item.setdefault('column_relevance', {})
            all_examples.append(item)
            project_examples[project_name].append(item)

    if VERBOSE:
        print(f"\n   总样本: {len(all_examples)}, 项目数: {len(project_examples)}")

    if sampling_strategy == 'specific_projects':
        if specific_projects is None or len(specific_projects) == 0:
            raise ValueError("需要指定 specific_projects")
        
        train_data = []
        remaining_data = []
        
        for project_name, examples in project_examples.items():
            if project_name in specific_projects:
                train_data.extend(examples)
            else:
                remaining_data.extend(examples)
        
        if len(train_data) > 0:
            val_size_in_train = val_ratio / (train_ratio + val_ratio)
            train_data, val_data = train_test_split(
                train_data, test_size=val_size_in_train,
                random_state=random_seed, shuffle=True
            )
        else:
            val_data = []
        
        test_data = remaining_data
        
    elif sampling_strategy == 'random_sample':
        if sample_ratio is None:
            raise ValueError("需要指定 sample_ratio")
        
        sample_size = int(len(all_examples) * sample_ratio)
        sampled_indices = random.sample(range(len(all_examples)), sample_size)
        remaining_indices = [i for i in range(len(all_examples)) if i not in sampled_indices]
        
        sampled_data = [all_examples[i] for i in sampled_indices]
        remaining_data = [all_examples[i] for i in remaining_indices]
        
        val_size_in_sampled = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            sampled_data, test_size=val_size_in_sampled,
            random_state=random_seed, shuffle=True
        )
        
        test_data = remaining_data
        
    else:
        train_data, temp_data = train_test_split(
            all_examples, test_size=(val_ratio + test_ratio),
            random_state=random_seed, shuffle=True
        )
        val_size_in_temp = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size_in_temp),
            random_state=random_seed, shuffle=True
        )

    print(f"   ✅ 训练集: {len(train_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")
    
    return train_data, val_data, test_data, all_examples


def load_full_dataset(data_file_path):
    
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_examples = []
    for project_name, project_data in data.items():
        for item in project_data:
            item['project_name'] = project_name
            if 'query' not in item:
                continue
            item.setdefault('query_related_table', [])
            item.setdefault('table_relevance', {})
            item.setdefault('column_relevance', {})
            all_examples.append(item)
    
    print(f"   ✅ 总样本数: {len(all_examples)}")
    return all_examples, data


# ==================== 主函数 ====================
def main():
    
    data_file_path = 'path/to/dataset_json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据采样配置
    sampling_strategy = 'random_sample'
    sample_ratio = 0.8
    # specific_projects = []
    
    # 加载数据
    train_data, val_data, test_data, all_examples = load_and_split_data(
        data_file_path,
        sampling_strategy=sampling_strategy,
        sample_ratio=sample_ratio,
        # specific_projects=specific_projects,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    # 初始化模型
    model_path = 'path/to/roberta-base'
    model = ContrastiveSchemaLinker(model_path, embedding_dim=256)

    if RUN_MODE == 'train':
        train_dataset = SinglePairDataset(train_data, model.tokenizer, max_length=512, pos_neg_balance=True)
        val_dataset = SinglePairDataset(val_data, model.tokenizer, max_length=512, pos_neg_balance=True)

        batch_size = 8
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        manual_threshold = None

        trainer = ImprovedSinglePairTrainer(
            model, train_dataloader, val_dataloader, val_data,
            test_data=test_data, device=device, temperature=0.07, 
            manual_threshold=manual_threshold, lr=2e-5
        )

        trainer.train(epochs=10)

        try:
            best_checkpoint = torch.load(
                os.path.join(SAVE_DIR, 'final_singlepair_model_with_threshold.pth'),
                weights_only=False
            )
            model.load_state_dict(best_checkpoint['model_state_dict'])
            trainer.optimal_threshold = best_checkpoint.get('optimal_threshold', 0.5)
            trainer.manual_threshold = best_checkpoint.get('manual_threshold', None)
            print(f"   ✅ 模型加载成功，阈值: {trainer.optimal_threshold:.4f}")
        except Exception as e:
            print(f"   ⚠️  使用当前模型")

        final_results, final_detailed, full_data_with_pred = trainer.inference_with_threshold(
            test_data, 
            use_top_k_percent=True,
            top_k_percent=0.5,
            save_detailed_results=True,
            return_full_data=True,
            original_data=test_data
        )

        final_precision = np.mean([r['precision'] for r in final_results]) if final_results else 0.0
        final_recall = np.mean([r['recall'] for r in final_results]) if final_results else 0.0
        final_f1 = np.mean([r['f1'] for r in final_results]) if final_results else 0.0

        final_result_summary = {
            'sampling_strategy': sampling_strategy,
            'sample_ratio': sample_ratio,
            # 'specific_projects': specific_projects,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'method': 'single_pair_binary_top_k',
            'selection_method': 'top_k_percent',
            'top_k_percent': 0.5,
            'final_metrics': {
                'precision': float(final_precision),
                'recall': float(final_recall),
                'f1': float(final_f1)
            },
            'total_test_samples': len(test_data)
        }
        result_file = os.path.join(SAVE_DIR, 'singlepair_top40_final_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result_summary, f, ensure_ascii=False, indent=2)

        if TEST_FULL_DATASET:
            full_dataset = train_data + val_data + test_data
            
            full_results, full_detailed, full_data_complete = trainer.inference_with_threshold(
                full_dataset,
                use_top_k_percent=True,
                top_k_percent=0.5,
                save_detailed_results=True,
                return_full_data=True,
                original_data=all_examples
            )
            
            full_precision = np.mean([r['precision'] for r in full_results]) if full_results else 0.0
            full_recall = np.mean([r['recall'] for r in full_results]) if full_results else 0.0
            full_f1 = np.mean([r['f1'] for r in full_results]) if full_results else 0.0
            
            full_result_by_project = defaultdict(list)
            for item in full_data_complete:
                project_name = item.get('project_name', 'unknown')
                full_result_by_project[project_name].append(item)
            
            full_result_file = os.path.join(SAVE_DIR, 'full_dataset_with_predictions.json')
            with open(full_result_file, 'w', encoding='utf-8') as f:
                json.dump(dict(full_result_by_project), f, ensure_ascii=False, indent=2)
            
            # 保存详细结果
            full_detailed_file = os.path.join(SAVE_DIR, 'full_dataset_detailed_results.json')
            with open(full_detailed_file, 'w', encoding='utf-8') as f:
                json.dump(full_detailed, f, ensure_ascii=False, indent=2)
            
            # 保存汇总统计
            full_summary = {
                'total_samples': len(full_dataset),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'method': 'single_pair_binary_top_k',
                'selection_method': 'top_k_percent',
                'top_k_percent': 0.5,
                'overall_metrics': {
                    'precision': float(full_precision),
                    'recall': float(full_recall),
                    'f1': float(full_f1)
                }
            }
            full_summary_file = os.path.join(SAVE_DIR, 'full_dataset_summary.json')
            with open(full_summary_file, 'w', encoding='utf-8') as f:
                json.dump(full_summary, f, ensure_ascii=False, indent=2)
    else:
        
        if not os.path.exists(TEST_MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {TEST_MODEL_PATH}")
        
        checkpoint = torch.load(TEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        manual_threshold = checkpoint.get('manual_threshold', None)
        temperature = checkpoint.get('temperature', 0.07)
        
        dummy_dataset = SinglePairDataset([], model.tokenizer, max_length=512)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=16)
        
        trainer = ImprovedSinglePairTrainer(
            model, dummy_dataloader, dummy_dataloader, [],
            test_data=test_data, device=device, temperature=temperature,
            manual_threshold=manual_threshold, lr=2e-5
        )
        trainer.optimal_threshold = optimal_threshold
        
        # 测试整个数据集（如果启用）
        if TEST_FULL_DATASET:
            
            # 加载完整数据集
            all_examples, original_data_dict = load_full_dataset(data_file_path)
            
            # 在整个数据集上推理
            full_results, full_detailed, full_data_complete = trainer.inference_with_threshold(
                all_examples,
                use_top_k_percent=True,
                top_k_percent=0.5,
                save_detailed_results=True,
                return_full_data=True,
                original_data=all_examples
            )
            
            full_precision = np.mean([r['precision'] for r in full_results]) if full_results else 0.0
            full_recall = np.mean([r['recall'] for r in full_results]) if full_results else 0.0
            full_f1 = np.mean([r['f1'] for r in full_results]) if full_results else 0.0
            
            full_result_by_project = defaultdict(list)
            for item in full_data_complete:
                project_name = item.get('project_name', 'unknown')
                full_result_by_project[project_name].append(item)
            
            full_result_file = os.path.join(SAVE_DIR, 'full_dataset_with_predictions.json')
            with open(full_result_file, 'w', encoding='utf-8') as f:
                json.dump(dict(full_result_by_project), f, ensure_ascii=False, indent=2)
            
            full_detailed_file = os.path.join(SAVE_DIR, 'full_dataset_detailed_results_80.json')
            with open(full_detailed_file, 'w', encoding='utf-8') as f:
                json.dump(full_detailed, f, ensure_ascii=False, indent=2)
            
            full_summary = {
                'mode': 'test_only',
                'model_path': TEST_MODEL_PATH,
                'total_samples': len(all_examples),
                'method': 'single_pair_binary_top_k',
                'selection_method': 'top_k_percent',
                'top_k_percent': 0.5,
                'threshold': float(optimal_threshold),
                'temperature': float(temperature),
                'overall_metrics': {
                    'precision': float(full_precision),
                    'recall': float(full_recall),
                    'f1': float(full_f1)
                }
            }
            full_summary_file = os.path.join(SAVE_DIR, 'full_dataset_summary.json')
            with open(full_summary_file, 'w', encoding='utf-8') as f:
                json.dump(full_summary, f, ensure_ascii=False, indent=2)
            
        else:
            final_results, final_detailed, test_data_with_pred = trainer.inference_with_threshold(
                test_data, 
                use_top_k_percent=True,
                top_k_percent=0.8,
                save_detailed_results=True,
                return_full_data=True,
                original_data=test_data
            )

            final_precision = np.mean([r['precision'] for r in final_results]) if final_results else 0.0
            final_recall = np.mean([r['recall'] for r in final_results]) if final_results else 0.0
            final_f1 = np.mean([r['f1'] for r in final_results]) if final_results else 0.0

            test_result_summary = {
                'mode': 'test_only',
                'model_path': TEST_MODEL_PATH,
                'test_samples': len(test_data),
                'method': 'single_pair_binary_top_k',
                'selection_method': 'top_k_percent',
                'top_k_percent': 0.5,
                'threshold': float(optimal_threshold),
                'temperature': float(temperature),
                'test_metrics': {
                    'precision': float(final_precision),
                    'recall': float(final_recall),
                    'f1': float(final_f1)
                }
            }
            result_file = os.path.join(SAVE_DIR, 'test_only_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(test_result_summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

def run_singlepair_by_paths(
    data_file_path: str,
    model_path: str,
    save_dir: str,
    test_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    端到端入口（只需路径），其它配置完全沿用你原脚本的默认值。
    新增返回: artifacts['contrastive_results_json'] -> detailed_results_top50_singlepair.json 的路径
    """
    global SAVE_DIR, TEST_MODEL_PATH
    SAVE_DIR = save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    if test_model_path is not None:
        TEST_MODEL_PATH = test_model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampling_strategy = 'random_sample'
    sample_ratio = 0.8
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    # --- 加载与切分数据 ---
    train_data, val_data, test_data, all_examples = load_and_split_data(
        data_file_path=data_file_path,
        sampling_strategy=sampling_strategy,
        sample_ratio=sample_ratio,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=42
    )

    # --- 初始化模型 ---
    model = ContrastiveSchemaLinker(model_path, embedding_dim=256)

    artifacts = {
        "save_dir": SAVE_DIR,
        "best_model": os.path.join(SAVE_DIR, 'best_singlepair_model_final.pth'),
        "final_with_threshold": os.path.join(SAVE_DIR, 'final_singlepair_model_with_threshold.pth'),
        "train_curves": os.path.join(SAVE_DIR, 'singlepair_training_curves_final.png'),
        # 关键产物：DCG详细结果（供后续 LLM 两步筛选使用）
        "contrastive_results_json": os.path.join(SAVE_DIR, 'detailed_results_top50_singlepair.json'),
    }
    summary: Dict[str, Any] = {}

    # ==================== 训练分支（沿用原逻辑，RUN_MODE 由你全局控制） ====================
    if RUN_MODE == 'train':
        train_dataset = SinglePairDataset(train_data, model.tokenizer, max_length=512, pos_neg_balance=True)
        val_dataset = SinglePairDataset(val_data, model.tokenizer, max_length=512, pos_neg_balance=True)

        batch_size = 8
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        manual_threshold = None
        trainer = ImprovedSinglePairTrainer(
            model, train_dataloader, val_dataloader, val_data,
            test_data=test_data, device=device, temperature=0.07,
            manual_threshold=manual_threshold, lr=2e-5
        )

        trainer.train(epochs=10)

        # 训练后按原逻辑加载最终模型
        try:
            best_checkpoint = torch.load(
                artifacts["final_with_threshold"],
                weights_only=False
            )
            model.load_state_dict(best_checkpoint['model_state_dict'])
            trainer.optimal_threshold = best_checkpoint.get('optimal_threshold', 0.5)
            trainer.manual_threshold = best_checkpoint.get('manual_threshold', None)
            print(f"   ✅ 模型加载成功，阈值: {trainer.optimal_threshold:.4f}")
        except Exception:
            print("   ⚠️  使用当前模型")

        # 在测试集推理（会写 detailed_results_top50_singlepair.json 到 SAVE_DIR）
        final_results, final_detailed, full_data_with_pred = trainer.inference_with_threshold(
            test_data,
            use_top_k_percent=True,
            top_k_percent=0.5,
            save_detailed_results=True,
            return_full_data=True,
            original_data=test_data
        )

        final_precision = float(np.mean([r['precision'] for r in final_results])) if final_results else 0.0
        final_recall   = float(np.mean([r['recall'] for r in final_results])) if final_results else 0.0
        final_f1       = float(np.mean([r['f1'] for r in final_results])) if final_results else 0.0

        result_file = os.path.join(SAVE_DIR, 'singlepair_top40_final_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sampling_strategy': sampling_strategy,
                'sample_ratio': sample_ratio,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'method': 'single_pair_binary_top_k',
                'selection_method': 'top_k_percent',
                'top_k_percent': 0.5,
                'final_metrics': {
                    'precision': final_precision,
                    'recall': final_recall,
                    'f1': final_f1
                },
                'total_test_samples': len(test_data)
            }, f, ensure_ascii=False, indent=2)

        # （可选）全量数据集推理（会再次写 detailed_results_top50_singlepair.json；如开启 TEST_FULL_DATASET）
        if TEST_FULL_DATASET:
            full_dataset = train_data + val_data + test_data
            full_results, full_detailed, full_data_complete = trainer.inference_with_threshold(
                full_dataset,
                use_top_k_percent=True,
                top_k_percent=0.5,
                save_detailed_results=True,   # 会覆盖同名 detailed_results_top50_singlepair.json
                return_full_data=True,
                original_data=all_examples
            )
            full_detailed_file = os.path.join(SAVE_DIR, 'full_dataset_detailed_results.json')
            with open(full_detailed_file, 'w', encoding='utf-8') as f:
                json.dump(full_detailed, f, ensure_ascii=False, indent=2)

        summary = {
            "mode": "train",
            "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
            "contrastive_results_json": artifacts["contrastive_results_json"],
        }

    # ==================== 测试分支（沿用原逻辑） ====================
    else:
        if not os.path.exists(TEST_MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {TEST_MODEL_PATH}")

        checkpoint = torch.load(TEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        optimal_threshold = float(checkpoint.get('optimal_threshold', 0.5))
        manual_threshold  = checkpoint.get('manual_threshold', None)
        temperature       = float(checkpoint.get('temperature', 0.07))

        dummy_dataset = SinglePairDataset([], model.tokenizer, max_length=512)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=16)

        trainer = ImprovedSinglePairTrainer(
            model, dummy_dataloader, dummy_dataloader, [],
            test_data=test_data, device=device, temperature=temperature,
            manual_threshold=manual_threshold, lr=2e-5
        )
        trainer.optimal_threshold = optimal_threshold

        # 跑推理，生成 detailed_results_top50_singlepair.json
        _ = trainer.inference_with_threshold(
            test_data,
            use_top_k_percent=True,
            top_k_percent=0.8 if not TEST_FULL_DATASET else 0.5,  # 与原 main 保持一致
            save_detailed_results=True,
            return_full_data=False
        )

        # 如果还开启了全量
        if TEST_FULL_DATASET:
            all_examples_full, _ = load_full_dataset(data_file_path)
            _ = trainer.inference_with_threshold(
                all_examples_full,
                use_top_k_percent=True,
                top_k_percent=0.5,
                save_detailed_results=True,   # 仍会覆盖 detailed_results_top50_singlepair.json
                return_full_data=False
            )

        summary = {
            "mode": "test",
            "contrastive_results_json": artifacts["contrastive_results_json"],
        }

    return {"summary": summary, "artifacts": artifacts}