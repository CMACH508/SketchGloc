
# Self-Supervised Learning for Enhancing Spatial Awareness in Free-Hand Sketch
![model](https://github.com/836790345/123/blob/main/model.pdf)
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
python train_reorganization.py --log_root=checkpoint_path --data_dir=dataset_path --resume_training=False --hparams="categories=[bee,bus]"
```

## Generating
```
python sample.py sample_size dataset_id sample_output re_scale checkpoint_path 0
```

## Evaluation

The metrics **Rec** and **Ret** are used to testify whether a method learns accurate and robust sketch representations.
```
python retrieval.py sample_input_path checkpoint_path
```
For calculating Rec, you need to train a Sketch_a_net for each dataset as the classifier. 
![metric](https://github.com/836790345/123/blob/main/metric.png)

## Hyperparameters
```

```
We also provide three pre-trained models used in the article, and you can get them from [link](https://quickdraw.withgoogle.com/data).


