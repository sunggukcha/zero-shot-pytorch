
# Zero-shot Learning PyTorch Implementation

  

Zero-shot classification, segmentation implementation in PyTorch

Supporting (re-)implementation

\[1] [SPNet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xian_Semantic_Projection_Network_for_Zero-_and_Few-Label_Semantic_Segmentation_CVPR_2019_paper.pdf) 

  

## Zero-shot Learning?

  

There is a field called 'few shot learning', in which only few number of training samples are given.

Zero-shot learning is a recognition field, which predicts *unseen* classes without a positive sample.


  

## Word-Embedding as External Knowledge

In order to recognize an unseen class, we need any external knowledge about the unseen class. Recently many works (e.g., \[1]) leverages word-embedding as external knowledge. In this repository, we use (Wikipedia or Common Crawl) pretrained Fasttext \[2].

  
  

## Zero-Shot Classification

All classification models supported by [Torchvision](https://pytorch.org/docs/stable/torchvision/models.html) is available (e.g., ResNet).

  

## Zero-Shot Segmentation

Deep base ResNet + DeeplabV2 + SPNet is supported.

  


### References

\[1]: http://openaccess.thecvf.com/content_CVPR_2019/papers/Xian_Semantic_Projection_Network_for_Zero-_and_Few-Label_Semantic_Segmentation_CVPR_2019_paper.pdf

\[2]: https://fasttext.cc/docs/en/english-vectors.html