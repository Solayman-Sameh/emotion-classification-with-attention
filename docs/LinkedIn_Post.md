# LinkedIn draft

I built an end-to-end emotion classification project as part of Instant AI & Data Analysis Training.

The app predicts six emotions from English text: joy, sadness, anger, fear, surprise, and disgust. I compared LSTM, GRU, BiLSTM with additive attention, and fine-tuned DistilBERT on the same GoEmotions-derived dataset.

DistilBERT led the comparison with 80.76% test accuracy and 0.7179 macro F1 across 3,357 held-out examples. Model selection used validation macro F1 before evaluating the test results.

The Gradio app pairs DistilBERT's predictions with a separately labeled BiLSTM attention view. It shows that model's own prediction and token weights, including when the two models disagree.

The work included label mapping, duplicate removal, training-only vocabulary and class weights, pretrained GloVe embeddings, per-class error analysis, and automatic retrieval of trained models from Hugging Face.

One useful lesson: headline accuracy does not tell the whole story. Minority-class results and the limits of attention visualizations matter too.

Demo: attach the finished recording to this post.

Code: https://github.com/Solayman-Sameh/emotion-classification-with-attention

Models: https://huggingface.co/Solayman-Sameh/emotion-classification-with-attention

#MachineLearning #NLP #DeepLearning #PyTorch #HuggingFace #Gradio
