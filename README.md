
# Self-Supervised Learning for Enhancing Spatial Awareness in Free-Hand Sketch
![model](https://github.com/CMACH508/SketchGloc/blob/main/imgs/model.pdf)
Free-hand sketch, as a versatile medium of communication, can be viewed as a collection of strokes arranged in a spatial layout to convey a concept. Due to the abstract nature of the sketches, changes in stroke position may make them difficult to recognize. Recently, Graphic sketch representations are effective in representing sketches. However, existing methods overlook the significance of the spatial layout of strokes and the phenomenon of strokes being drawn in the wrong positions is common. Therefore, we developed a self-supervised task to correct stroke placement and investigate the impact of spatial layout on learning sketch representations. For this task, we propose a spatially aware method, named SketchGloc, utilizing multiple graphs for graphic sketch representations. This method utilizes grids for each stroke to describe the spatial layout with other strokes, allowing for the construction of multiple graphs. Unlike other methods that rely on a single graph, this design conveys more detailed spatial layout information and alleviates the impact of misplaced strokes. The experimental results demonstrate that our model outperforms existing methods in both our proposed task and the traditional controllable sketch synthesis task. Additionally, we found that SketchGloc can learn more robust representations under our proposed task setting. 

## Dataset

You need the original sketch sequences from [QuickDraw dataset](https://quickdraw.withgoogle.com/data)



## Required environments

1. Python 3.6
2. Tensorflow 1.15
   
   ```pip3 install -r requirements.txt```

## Training
```
python train.py --log_root=checkpoint_path --data_dir=dataset_path --resume_training=False --hparams="categories=[bee,bus]"
```

## Generating
```
python sample_abs.py sample_size dataset_id sample_output re_scale checkpoint_path 0
```

## Evaluation

The metrics **Rec** and **Ret** are used to testify whether a method learns accurate and robust sketch representations.
```
python retrieval.py sample_input_path checkpoint_path
```
For calculating Rec, you need to train a Sketch_a_net for each dataset as the classifier. 
![metric](https://github.com/CMACH508/SketchGloc/blob/main/imgs/metric.png)

## Hyperparameters
```
categories=['pig','bee','flower','bus','giraffe','car', 'cat' , 'horse'],  
        # Sketch categories 'pig','bee','flower','bus','giraffe'
        #ds2: 'airplane', 'angel', 'apple', 'butterfly', 'bus', 'cake','fish', 'spider', 'The Great Wall','umbrella'
        #ds3:'pig','bee','flower','bus','giraffe','car', 'cat' , 'horse'
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the loss is not improved)
        re_scale=0, #Reorganization task setting
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=512,  # Size of decoder
        enc_model='lstm',
        enc_rnn_size=512,
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        max_stroke_len = -1,
        max_strokes_num = -1,
        z_size=128,  # Size of latent variable
        batch_size=100,  # Minibatch size
        num_mixture=5,  # Recommend to set to the number of categories
        learning_rate=1e-3,  # Learning rate
        decay_rate=0.99999,   # Learning rate decay per minibatch.
        min_learning_rate=1e-5,  # Minimum learning rate
        grad_clip=1.,  # Gradient clipping
        de_weight=1.,  # Weight for deconv loss
        use_recurrent_dropout=True,  # Dropout with memory loss
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout
        input_dropout_prob=0.9,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput
        output_dropout_prob=0.9,  # Probability of output dropout keep
        random_scale_factor=0.1,  # Random scaling data augmention proportion
        augment_stroke_prob=0.1,  # Point dropping augmentation proportion
        is_training=True,  # Training mode or not
        num_per_category=70000  # Training samples from each category
```
We also provide three pre-trained models used in the article, and you can get them from [link](https://quickdraw.withgoogle.com/data).


