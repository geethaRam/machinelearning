# Sequence Models

## Introduction
* Sequence models  use recurrent neural networks (RNN) where input or output is a sequence data 
* Examples of Sequence models:
** Speech recognition : Audio to text. Audio is sequence data
** Music generation : Output is sequence data (music)
** Sentiment classification : Input text is sequence data
** DNA Sequence analysis
** Machine Translation
** Video activity recognition
** Name entity recognition

## Notations
Input: x : harry potter met hermione granger met at the hallway

Goal : Identify name entities (Harry potter and Hermione granger)

output y:  1      1      0     1        1     0   0   0  0

Ouput is the binary representation if the given word is a named entity or not

Each word in input sentence is x<sup>(i)</sup>. So, x<sup>(1)</sup>,x<sup>(2)</sup>....x<sup>(9)</sup>
Same for output format y as well.
The total number of words in the sentence is T<sub>x</sub>=9 and similary for the output T<sub>y</sub>

Now, the training example can have many sentences / many examples in the training set. 
So, if we have i training examples then the notations would be

Input : X<sup>(i)<t></sup>
output : Y<sup>(i)<t></sup>
  Number of words in input : T<sub>x</sub><sup>(i)</sup>
  Number of words in ouput : T<sub>y</sub><sup>(i)</sup>

## Representing words in Vocabulary

Define a vocabulary vector of 10,000(for eg) words. Commercial tools use 50,000 or 60,000 words in the vocabulary matrix.

For a given sentence like : harry potter met hermione granger met at the hallway
We make 9 vectors for each words. Each vector will have 10,000 rows. The value of the row number - that matches the word in the input with that of the vocabulary vector will be 1. The rest of the values will be 0.

So, if the vocabulary vector had Harry in 650th row, then the first word vector for the input sentence will have the 650th row value as 1 and the rest of the row values as 0.
Continue to define the vectors for each word in the sentence.

## Recurrent Neural Network (RNN)

### Why not use standard neural network?
 * A standard neural network has same number of inputs and outputs across all training data. But in NLP problems, it is not guaranteed, Each sentence can have different number of words
 * Doesnt share features learned across different positions of text
 
 ### RNN
 * Each layer in the NN will use data from the previous layer.
 * Mostly tanh/relu is used for the activiation function on the Input & intermediate layers
 * The activation function for the final layer will be sigmoid (if it is binary classification) or softmax( if it is prediction)
 * Simplified RNN Notation 
  ** Instead of carrying forward the two matrices, W<sub>aa</sub> and W<sub>ax</sub> - we can compress both of them together into a single matrix so that we have only one matrix to use in our computations. However the shape of this new matrix will be combination of the shapes of these two individual matrices 
  ![Reference](https://github.com/geethaRam/machinelearning/blob/master/simple_rnn.png "Logo Title Text 1")





