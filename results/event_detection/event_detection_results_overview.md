## EstBERT events model training results

| model name | train_eval_recall | train eval f1 | final_test_recall | final test f1 | pred tempf news recall | pred tempf rkogu recall | pred tempf horisont recall
| --- | --- | --- | --- | --- | --- | --- | --- 
|EstBERT_no_classes_all| 0.9 | 0.88 | 0.86 | **0.86** | 0.79 | 0.64 | 0.66
|EstBERT_no_classes_main| 0.93 | **0.91** | 0.85 | 0.85 | **0.81** | **0.66** | **0.67**
|EstBERT_w_classes_all| 0.74 | 0.72 | 0.7 | 0.70 | 0.61 | 0.36 | 0.5
|EstBERT_w_classes_main| 0.75 | 0.73 | 0.74 | 0.72 | 0.62 | 0.41 | 0.5
|EstBERT_w_agg_classes_all| 0.76 | 0.76 | 0.71 | 0.72 | 0.6 | 0.41 | 0.48
|EstBERT_w_agg_classes_main| 0.81 | 0.8 | 0.76 | 0.76 | 0.62 | 0.39 | 0.49
|EstBERT_w_agg_classes_no_modal_all| 0.78| 0.74 | 0.74 | 0.74 | 0.64 | 0.39 | 0.69
|EstBERT_w_agg_classes_no_modal_main| 0.78 | 0.74 | 0.76 | 0.74 | 0.63 | 0.44 | 0.61

## Est-RoBERTa events model training results

| model name | train_eval_recall | train eval f1 | final_test_recall | final test f1 | pred tempf news recall | pred tempf rkogu recall | pred tempf horisont recall
| --- | --- | --- | --- | --- | --- | --- | ---
|Est-RoBERTa_no_classes_all| 0.92 | 0.89 | 0.9 | 0.86 | **0.83** | **0.77** | **0.82**
|Est-RoBERTa_no_classes_main| 0.94 | **0.93** | 0.89 | **0.88** | 0.81 | 0.73 | 0.82
|Est-RoBERTa_w_classes_all| 0.78 | 0.77 | 0.77 | 0.74 | 0.65 | 0.49 | 0.64
|Est-RoBERTa_w_classes_main| 0.8 | 0.8 | 0.76 | 0.76 | 0.64 | 0.51 | 0.63
|Est-RoBERTa_w_agg_classes_all| 0.82 | 0.81 | 0.77 | 0.75 | 0.66 | 0.49 | 0.73
|Est-RoBERTa_w_agg_classes_main| 0.82 | 0.83 | 0.75 | 0.76 | 0.63 | 0.45 | 0.63
|Est-RoBERTa_w_agg_classes_no_modal_all| 0.78 | 0.78 | 0.75 | 0.74 | 0.64 | 0.47 | 0.68
|Est-RoBERTa_w_agg_classes_no_modal_main| 0.83 | 0.82 | 0.81 | 0.79 | 0.67 | 0.52 | 0.63
