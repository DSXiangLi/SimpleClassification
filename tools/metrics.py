# -*-coding:utf-8 -*-
"""
    二分类/多分类 TF Metrics & Evaluation Report
"""
import tensorflow as tf
import pandas as pd
from tensorboard import summary
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score


def pr_summary_hook(probs, labels, num_threshold, output_dir, save_steps):
    pr_summary = summary.pr_curve( name='pr_curve',
                                   predictions=probs[:, 1],
                                   labels=tf.cast(labels, tf.bool),
                                   num_thresholds=num_threshold )

    summary_hook = tf.train.SummarySaverHook(
        save_steps= save_steps,
        output_dir= output_dir,
        summary_op=[pr_summary]
    )
    return summary_hook


def binary_cls_metrics(probs, labels):
    """
    二分类任务 TF Metrics
        probs: (n_samples, 2)
        labels: (n_samples,)
    """
    predictions = tf.argmax(probs, axis=-1)
    precision, precision_op = tf.metrics.precision(labels, predictions=predictions)
    recall, recall_op = tf.metrics.recall(labels, predictions=predictions)
    f1 = 2 * (precision * recall) / (precision + recall)

    eval_metric_ops = {
        'metrics/accuracy': tf.metrics.accuracy(labels, predictions=predictions),
        'metrics/auc': tf.metrics.auc(labels, predictions=probs[:, 1], curve='ROC',
                                      summation_method='careful_interpolation'),
        'metrics/pr': tf.metrics.auc(labels, predictions=probs[:, 1], curve='PR',
                                     summation_method='careful_interpolation'),
        'metrics/precision': (precision, precision_op),
        'metrics/recall': (recall, recall_op),
        'metrics/f1': (f1, tf.identity(f1))
    }
    return eval_metric_ops


def binary_cls_report(probs, labels, thresholds):
    """
    二分类任务Evaluation
        probs: (n_samples, 2)
        labels: (n_samples,)
        threhosld: 计算不同阈值下的precision，recall和f1
    """
    probs = [i[1] for i in probs]
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    n_sample = len(probs)
    n_pos = sum(labels)
    # Precision & Recall by threshold
    result = []
    for thr in thresholds:
        tmp = [int(i > thr) for i in probs]
        precision = precision_score(labels, tmp)
        recall = recall_score(labels, tmp)
        accuracy = accuracy_score(labels, tmp)
        result.append((thr, sum(tmp), precision, recall, accuracy, auc, ap, n_sample, n_pos))

    df = pd.DataFrame(result, columns=['threshold', 'n', 'precision', 'recall', 'accuracy', 'auc', 'ap','total','total_pos'])
    df = df.to_string(formatters={'threhsold': "{:.2f}".format,
                                  'n': "{0:d}".format, 'precision': "{:.1%}".format,
                                  'recall': "{:.1%}".format, 'accuracy': "{:.1%}".format,
                                  'auc': "{:.1%}".format,'ap': "{:.1%}".format,
                                  'total': '{0:d}'.format, 'total_pos': '{0:d}'.format
                                  })
    return df


def multi_cls_metrics(probs, labels, idx2label):
    """
    多分类任务 TF Metrics
        probs: (n_samples, 2)
        labels: (n_samples,)
        idx2label: labelid 到分类名称的映射
    支持
    1. Overall Accuracy, AUC, AP
    2. 分label的precision， recall，f1
    3. micro, macro: precision, recall, f1
    """
    num_labels = len(idx2label)
    predictions = tf.argmax(probs, axis=-1)
    metric_ops = {
        'metrics/overall_accuracy': tf.metrics.accuracy(labels, predictions),
        'metrics/overall_auc': tf.metrics.auc(labels, predictions=probs[:, 1], curve='ROC',
                                      multi_label=True, num_labels = num_labels,
                                      summation_method='careful_interpolation'),
        'metrics/overall_pr': tf.metrics.auc(labels, predictions=probs[:, 1], curve='PR',
                                     multi_label=True, num_labels=num_labels,
                                     summation_method='careful_interpolation')
    }
    recalls = []
    precisions = []
    recall_ops = []
    precision_ops = []
    for idx in range(num_labels):
        label_l = tf.equal(labels, idx)
        prediction_l = tf.equal(predictions, idx)
        recall, recall_op = tf.metrics.recall(
            labels=label_l,
            predictions=prediction_l
        )
        precision, precision_op = tf.metrics.precision(
            labels=label_l,
            predictions=prediction_l
        )
        f1 = 2 * (precision * recall) / (precision + recall)
        metric_ops.update(
            {'metrics/{}_precision'.format(idx2label[idx]): (precision, precision_op),
             'metrics/{}_recall'.format(idx2label[idx]): (recall, recall_op),
             'metrics/{}_f1'.format(idx2label[idx]): (f1, tf.identity(f1))
             }
        )
        recalls.append(recall)
        precisions.append(precision)
        recall_ops.append(recall_op)
        precision_ops.append(precision_op)
    precision, precision_op = (sum(precisions)/num_labels, sum(precision_ops)/num_labels)
    recall, recall_op = (sum(recalls)/num_labels, sum(recall_ops)/num_labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    metric_ops.update({
        'metrics/micro_precision': (precision, precision_op),
        'metrics/micro_recall': (recall, recall_op),
        'metrics/micro_f1': (f1, tf.identity(f1))
    })
    return metric_ops


def multi_cls_report(probs, labels, idx2label):
    """
    多分类任务 Evaluation
        probs: (n_samples, 2)
        labels: (n_samples,)
        idx2label: labelid 到分类名称的映射
    支持
    1. Overall Accuracy, AUC, AP
    2. 分label的precision， recall，f1
    3. micro, macro: precision, recall, f1
    """
    return None
