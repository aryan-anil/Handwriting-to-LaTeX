# Handwriting-to-LaTeX

A deep learning model that converts handwritten mathematical expressions into LaTeX code. This project uses a Vision Transformer (ViT) as the encoder to process handwriting images, and a GPT-2 decoder with cross-attention to generate the corresponding LaTeX markup.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e9296967-f864-44aa-9407-062d08f4970e" alt="Input Handwriting" width="400"/>
  <img src="https://github.com/user-attachments/assets/1b59a78a-54a3-41e9-8733-662b7664f998" alt="Output LaTeX Code" width="400"/>
  <img src="https://github.com/user-attachments/assets/b630f11a-b50a-4202-8b94-3f743db03685" alt="Compiled LaTeX using online compiler"/>
</div>


##  Features

- Translates handwritten math expressions into LaTeX.
- Uses a ViT-GPT2 encoder-decoder architecture.
- Built on top of the [MathWriting](https://arxiv.org/abs/2404.10690) dataset.
- Trained using noise augmentation to improve robustness and generalization.

##  Model Architecture

- **Encoder**: Vision Transformer (ViT) for visual understanding of handwritten input.
- **Decoder**: GPT-2 with cross-attention to translate image embeddings into LaTeX code.


## Dataset

The model is trained on the **MathWriting** dataset, which contains a large collection of handwritten mathematical expressions paired with their corresponding LaTeX annotations.

**Data Augmentation**:  
To improve generalization, noise was added to the input images (e.g., Gaussian noise, blur) prior to training. This augmentation simulates real-world handwritten imperfections and boosts performance on noisy input.

## Dependencies
- Python 3.8+
- PyTorch
- Transformers
- OpenCV
- PIL

## Citation
```
@article{gervais2024mathwriting,
  title={Mathwriting: A dataset for handwritten mathematical expression recognition},
  author={Gervais, Philippe and Fadeeva, Asya and Maksai, Andrii},
  journal={arXiv preprint arXiv:2404.10690},
  year={2024}
}
```
