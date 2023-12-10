# Autism-detection-using-deep-learning

# Overview
The Autism Detection using Deep Learning project aims to leverage state-of-the-art deep learning techniques to classify emotions in children, specifically focusing on distinguishing between autistic and non-autistic children. The primary goal is to provide a tool that aids in the early detection of autism spectrum disorders by analyzing facial expressions associated with various emotions.

# Dataset
The dataset used for this project was obtained from Kaggle, provided by Dr. Fatma M. Talaat and named Autistic Children Emotions. This dataset encompasses emotions such as fear, joy, sadness, surprise, and anger. For non-autistic children, since a dedicated dataset was not readily available, web scraping was performed to collect relevant data. This data augmentation strategy allowed us to create a more balanced dataset for training the model.


# Model Architecture
The deep learning model employs the EfficientNet architecture, known for its exceptional efficiency and accuracy. We fine-tuned the model to classify emotions into two primary classes: autistic and non-autistic. The pre-trained weights on the emotion classes were extracted, and the model was retrained on the new binary classification task.

# Training and Evaluation
After retraining, the model achieved an impressive accuracy of 75% on the validation set. Subsequently, to evaluate the model's generalization capability, we tested it on another dataset obtained from Kaggle. In this evaluation, the model maintained a strong performance, achieving an accuracy of 64.7%.
