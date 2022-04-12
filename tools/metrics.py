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
                                   predictions=probs[:,1],
                                   labels=tf.cast( labels, tf.bool ),
                                   num_thresholds= num_threshold )

    summary_hook = tf.train.SummarySaverHook(
        save_steps= save_steps,
        output_dir= output_dir,
        summary_op=[pr_summary]
    )
    return summary_hook

def binary_cls_metrics(probs, labels, threshold):
    prediction = tf.to_float(tf.greater_equal(probs[:, 1], threshold))
    precision, precision_op = tf.metrics.precision(labels, predictions=prediction)
    recall, recall_op = tf.metrics.recall(labels, predictions=prediction)
    f1 = 2 * (precision * recall) / (precision + recall)

    eval_metric_ops = {
        'metrics/accuracy': tf.metrics.accuracy(labels, predictions=prediction),
        'metrics/auc': tf.metrics.auc(labels, predictions=probs[:1], curve='ROC',
                                      summation_method='careful_interpolation'),
        'metrics/pr': tf.metrics.auc(labels, predictions=probs[:1], curve='PR',
                                     summation_method='careful_interpolation'),
        'metrics/precision': (precision, precision_op),
        'metrics/recall': (recall, recall_op),
        'metrics/f1': (f1, tf.identity(f1))
    }
    return eval_metric_ops


def binary_cls_report(probs, labels, thresholds):
    """
    二分类任务Evaluation
        probs: (n_samples,)
        labels: (n_samples,)
        threhosld: 计算不同阈值下的precision，recall和f1
    """
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    # Precision & Recall by threshold
    result = []
    for thr in thresholds:
        tmp = [int(i > thr) for i in probs]
        precision = precision_score(labels, tmp)
        recall = recall_score(labels, tmp)
        accuracy = accuracy_score(labels, tmp)
        result.append((thr, sum(tmp), precision, recall, accuracy, auc, ap))

    df = pd.DataFrame(result, columns=['threshold', 'n', 'precision', 'recall', 'accuracy', 'auc', 'ap'])
    df = df.to_string(formatters={'threhsold': "{:.2f}".format,
                                  'n': "{:.0}".format, 'precision': "{:.1%}".format,
                                  'recall': "{:.1%}".format, 'accuracy': "{:.1%}".format,
                                  'auc': "{:.1%}".format,'ap': "{:.1%}".format
                                  })
    return df

def multi_cls_report(probs, labels, idx2tag):
    """
    多分类任务Evaluation
    probs： (n_samples, n_classes)
    labels:  (n_samples, n_classes)
    idx2tag: label到分类的
    """
    pass


def cls_report(probs, labels, idx2tag=None, threshold=None):
    """
    分类任务Evaluation Wrapper
    """
    if probs[0].shape ==2:
        # 二分类任务: probs和label
        probs = []
        pass