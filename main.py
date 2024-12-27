import matplotlib.pyplot as plt
import numpy as np
import torch as torch
from scipy import interp
from sklearn import metrics
from train import Train_loop


if __name__ == '__main__':
    #dgl.backend.load_backend('pytorch')
    #dgl.load_backend('pytorch')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # device = torch.device("cpu")

    lp_auc, lp_acc, lp_pre, lp_recall, lp_f1, lp_auprc, lp_fprs, lp_tprs = Train_loop(epochs=300,
                                                                hidden_size=2048,
                                                                dropout=0.2,
                                                                slope=0.1,  # LeakyReLU
                                                                lr=0.0015,
                                                                wd=1e-3,
                                                                random_seed=42,
                                                                device=device)


    print('-AUC LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_auc), np.std(lp_auc)),
          'Accuracy LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_acc), np.std(lp_acc)),
          'Precision LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_pre), np.std(lp_pre)),
          'Recall LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_recall), np.std(lp_recall)),
          'F1-score LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_f1), np.std(lp_f1)),
          'AUPRC LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_auprc), np.std(lp_auprc)),
          )


    lp_mean_fpr = np.linspace(0, 1, 10000)
    lp_tpr = []

    for i in range(len(lp_fprs)):
        lp_tpr.append(interp(lp_mean_fpr, lp_fprs[i], lp_tprs[i]))
        lp_tpr[-1][0] = 0.0
        plt.plot(lp_fprs[i], lp_tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, lp_auc[i]))

    lp_mean_tpr = np.mean(lp_tpr, axis=0)
    lp_mean_tpr[-1] = 1.0
    lp_mean_auc = metrics.auc(lp_mean_fpr, lp_mean_tpr)
    lp_auc_std = np.std(lp_auc)
    plt.plot(lp_mean_fpr, lp_mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (lp_mean_auc, lp_auc_std))

    lp_std_tpr = np.std(lp_tpr, axis=0)
    lp_tpr_upper = np.minimum(lp_mean_tpr + lp_std_tpr, 1)
    lp_tpr_lower = np.maximum(lp_mean_tpr - lp_std_tpr, 0)
    plt.fill_between(lp_mean_fpr, lp_tpr_lower, lp_tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Link Prediction ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
