# Emotion-Analysis-Using-Speech


Introduction 
 
 
When you are listening to music and the music player automatically plays songs matching to your emotions. This is one of the many use cases of Emotion detection using speech. 
 
Our main goal is to come up with a robust deep learning model which can accurately and efficiently classify the emotions from given audio. For this we have used two methods, one by directly analysing the speech and another by changing the speech into text.  
 
For the first one, we have used available dataset which contains around 1000, audio clips (5 seconds) which are labelled with corresponding emotions along with the gender. We created two different deep learning models using CNN and LSTM. As the main challenge was identification of different features in the speech. For this, we used Librosa library in Python. 
 
For the second one, we have used GoogleAPI for speech recognition and linear SVC for analysing emotion through text.   
 
 
 
Problem Statement 
 
 
To create a deep learning model to analyse the emotion of a person through speech which is applicable in various ecommerce and IT application. 
 
 
 
 
 
Why and how the project was chosen? 
 
 
 
Emotion detection has become one of the biggest marketing strategies, in which mood of the consumer plays an important role. 
So to detect the current emotion of the person and suggest him the apt product or help him accordingly, will increase the demand of the product or the company. 
One among so many examples is that the AI virtual assistant product of Amazon can use this motion detection to help the customer to play music according to his mood or play what he likes based on his mood. 
So the vast area of practical application and importance in the real world, made us choose this topic. 
 
 
 
 
 

Project Design 
 
 
    • Collected labelled data. 
    • Created a CNN model with 16 layers. 
    • Trained the model using 80% of available data 
    • Tested the model using 20% of data with a max accuracy of 60%. 
 
 

 
 
 
    • Setting parameters: like sampling rate, window size (choosing one audio from the file) 
    • Setting the labels of feelings. 
    • Getting the features of audio files using Librosa.  
    • Dividing the data into test and train. 
    • Building the model. 
    • Save model and evaluate it on test data. 
    • Test on real world samples. 
 
 

 
 
Functionality 
 
 
Emotion analysis can be used in various fields. Vault unlocking using voice can be made more accurate using emotion detection, virtual assistants can be made more user friendly-like playing music according to users emotion, reading assistant can read the text emotionally, ecommerce site can use emotion detection for recommendation and suggestions. 
 
 
 
 
 
 

 
 
Implementation Details 
 
 
Basically, we have used two different approaches, first one is directly analysing the audio and second one is by converting the speech to text and then analysing the text. 
 
For the first approach also we have designed two different models CNN and LSTM. In the neural network we have constructed there are 1 Input Layer, 17 Hidden Layer, 1 Output Layer, 2 Dropouts, Batchsize = 32 and 1000 Epochs. The activation function used are tanh, sigmoid and dense. For feature extraction from the audio we have used Librosa and for creating the models we have used Keras Framework. 
 
For the second approach we have first converted the speech into text by using available GoogleAPI which uses deep learning models. For analysing the emotion of the text we have used Linear SVC. 
 
The code was written in Python3.6 and for maintaining our project we used Jupyter Notebook. It was executed on NVIDIA DGX (8X Tesla V100) Supercomputer Servers.  
 
 
 
 

 
 
 

 
 
 
 
 
 
Project Development Time Schedule 
 
 
 

7th - 10th Dec 
11th - 13th Dec
13th - 24th Dec
22th - 25th Dec
25th - 28th Dec 
Literature Survey





Datasets Collection





Coding





Review





Documentation



Learning and Reflections 
 
 
In the project we have implemented some important deep learning models like CNN and LSTM. We used Librosa for feature extraction of audio and GoogleAPI for converting speech to text. While doing this project we came to know how to create CNN and LSTM models and how to train and test the models. We learnt how to improve the accuracy of the model by adjusting the hyper parameters.  
 
 

 
Conclusion 
 
This model can be used by various apps, online shopping websites and so on to know about the user’s emotions. Further improvements can be made to the model so that it can perform well in real time. For improving the accuracy of the model, we can increase the size of the dataset. The classifier can be embedded in a software or an app so that it can work in real time. Moreover, we look forward to come up with more industrial applications of this model. 
 
 
 
 

 
 

Limitations and Future Enhancements 
 
Due to less availability of datasets our accuracy was not as expected. So, in further work we can increase the number of datasets so as to get higher accuracy.  
 
As we have used two methods, one by directly analysing the speech and another by changing the speech into text. In this project we have displayed the results separately. We can combine the results from two method to get more accurate results. For combining, we can give weightage for both methods higher for the first one and lower for the second one. 
 
 

 
 
References 
 
 
    1. https://zenodo.org/record/1188976/?f=3#.XCRnEFUzbZs 
    2. http://adventuresinmachinelearning.com/keras-lstm-tutorial/ 
    3. https://www.udemy.com/nlp-natural-language-processing-with-python/    
    4. https://www.kaggle.com/c/sa-emotions 
    5. https://www.sciencedirect.com/science/article/pii/S1877050915021432 
    6. https://ieeexplore.ieee.org/document/7975169 




YouTube Video: https://www.youtube.com/watch?v=h8RBi6tiK0c&feature=youtu.be
