# Period-alternatives-of-Softmax
Experimental Demo for our paper 
# 'Escaping the Gradient Vanishing: Periodic Alternatives of Softmax in Attention Mechanism'
We suggest that replacing the exponential function by periodic functions.
Through experiments on a simply designed demo referenced to LeViT, our method is proved to be able to alleviate the gradient problem and yield substantial improvements compared to Softmax and its variants.

** Create your own 'dataset' fold, and maybe need to modify the demo.py file for your own dataset except for cifar-10, cifar-100 and Tiny-imageNet.
# Function available:
    softmax , norm_softmax
    sinmax, norm_sinmax
    cosmax, norm_cosmax
    sin_2_max, norm_sin_2_max
    sin_2_max_move, norm_sin_2_max_move
    sirenmax, norm_sirenmax
    sin_softmax, norm_sin_softmax
# mode available:
    search:
            Random search for a suitable set of learning rate and weight decay, and record the results in 
            Attention_test/*functions/lr_wd_search.txt
    run:
            Train the demo, and there will be four .npy files created in root.
            (1) 'record_val_acc.npy' for val acc record every 100 iter;
            (2) 'record_train_acc.npy' for train acc record every batch;
            (3) 'record_loss.npy' for train loss record every batch;
            (4) 'kq_value.npy' for Q.K record *before sclaled*.
    att_run:
            Same as the run mode but:
            (1) No kq_value record;
            (2) Every 5 epoch, input a test image and record the attention score map of each head of each layer.
                Saved in 'Attention_test/attention_maps.npy' 
