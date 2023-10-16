print("===> Evaluation on validation set: \n"
          "laneline F-measure {:.8} \n"
          "laneline Recall  {:.8} \n"
          "laneline Precision  {:.8} \n"
          "laneline Category Accuracy  {:.8} \n"
          "laneline x error (close)  {:.8} m\n"
          "laneline x error (far)  {:.8} m\n"
          "laneline z error (close)  {:.8} m\n"
          "laneline z error (far)  {:.8} m\n".format(eval_stats[0], eval_stats[1],
                                                     eval_stats[2], eval_stats[3],
                                                     eval_stats[4], eval_stats[5],
                                                     eval_stats[6], eval_stats[7]))