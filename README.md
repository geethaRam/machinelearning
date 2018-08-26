# Sequence Models

## Introduction
* Sequence models  use recurrent neural networks (RNN) where input or output is a sequence data 
* Examples of Sequence models:
  * Speech recognition : Audio to text. Audio is sequence data
  * Music generation : Output is sequence data (music)
  * Sentiment classification : Input text is sequence data
  * DNA Sequence analysis
  * Machine Translation
  * Video activity recognition
  * Name entity recognition

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

 Superscript  [l][l]  denotes an object associated with the  lthlth  layer.

 Example:  a[4]a[4]  is the  4th4th  layer activation.  W[5]W[5]  and  b[5]b[5]  are the  5th5th  layer parameters.
 Superscript  (i)(i)  denotes an object associated with the  ithith  example.

 Example:  x(i)x(i)  is the  ithith  training example input.
 Superscript  ⟨t⟩⟨t⟩  denotes an object at the  tthtth  time-step.

 Example:  x⟨t⟩x⟨t⟩  is the input x at the  tthtth  time-step.  x(i)⟨t⟩x(i)⟨t⟩  is the input at the  tthtth  timestep of         example  ii .
 Lowerscript  ii  denotes the  ithith  entry of a vector.

  Example:  a[l]iai[l]  denotes the  ithith  entry of the activations in layer  ll .

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
   * Instead of carrying forward the two matrices, W<sub>aa</sub> and W<sub>ax</sub> - we can compress both of them together        into a single matrix so that we have only one matrix to use in our computations. However the shape of this new matrix          will be   combination of the shapes of these two individual matrices 
  ![Reference](https://github.com/geethaRam/machinelearning/blob/master/simple_rnn.png "Logo Title Text 1")
  
 
 ## Backpropagation
* In order to compute backpropagation (similar to what we saw in Logistic regression), we need to define a loss function
* Find the loss associated with predicting a single word
* Then aggregate the whole loss to the entire sequence
* Based on the loss, we update the parameters in each back prop step using gradient descent
* In the below example, the green arrows show forward propagation. The blue arrows show Loss function calculation and the red arrows show backward propagation
 
   ![Reference](https://github.com/geethaRam/machinelearning/blob/master/back_rnn.png "Logo Title Text 1")
 
 
 
 ## Different types of RNNs
 * In the previous examples we saw, the input and ouptut size are equal. But it is not possible to always have the same input and ouput length for all problems in NLP
 * Many to Many architecture: (if input and output length are the same)
   * Many inputs and many outputs 
   * As we saw in previous examples
 * many to one architecture
   * In the case of a sentiment classification problem, there will be many input word vectors, however the final output has to be 0/1 or 1...5 - one of the sentiments. 
   * In this case, we dont have to compute the output y - in each input layer of the RNN - rather the final layer can calculate the sentiment of the entire sequence of the word vectors. hence this has many inputs and one output 
 * One-Many architecture
   * In the case of music generation, the input is usually a music type or no input at all but the output is many 
 * Many to Many architecture:(if input and output length are different)
   * Eg: machine translation. 
   * In this case, the input will be in a different language which will have a different set of words.
   * The input will then go through the encoder for processing and then into decoder that produces the output
 * One-One architecture
   *  this is just a standard NN. Not a RNN
 
 ## Sequence generation
 * Get a training set : large corpus of english text
 * Tokenize each sentences into vectors in the training set. 
 * While tokenizing, we add <EOS> end-of-sentence to determine the sentence-end
 * What to do when the word in the input is not in the corpus of text - you modify that word to <UNKN> unknown
 * Then apply the RNN model that takes input from the previous layer predictions for each word in the sequence
 * Use softmax as the activation function since we are determining the probability of a given word in the input matches in the training word vector.
 * In Sequence generation - you have an encoder to process the input and decoder to produce the output
 * Sample use cases: Machine translation, Question/Answering systems, Chatbots.
 
 ## Challenges in RNN
 * Lets see an example sequence: 
   * Sequence 1: The cat, which ate those foods at the event and at the party, was full.
   * Sequence 2: The cats, which ate ........................................, were full.
   * Notice in the first sequence, we need to determine that `was` is associated with `cat` and `were` is associated with          `cats`.
   * Now the output of `was` or `were` is from a NN layer that is much farther from the `cat` or `cats` NN layer. So, it is        very difficult for the gradient descent of `was` NN layer to have a meaningful impact on the `cat` layer. 
   * This is a common RNN problem in NLP when we create word-level Layers where it is possible that there may be LARGE number      of NN layer(upto 100 layers or more than that) when modeling an english sentence. 
   * This is called **vanishing gradients** where gradients decrease exponentially. 
   * There is also another problem called **Exploding gradients**
   * With larger NN - then gradients can increase exponentially. Because of this, the RNN model can set parameters to NaN (numerical overflow).
   * Solution: Gradient clipping - relatively robust solution
   * Vanishing gradient is difficult to solve compared to Exploding gradient.
   
  ## Gated Recurrent Units (GRU)
  * The equation for forward propagation of RNN
                
       a<sup>t</sup>=g(W<sub>a</sub>[a<sup>t-1</sup>,x<sup>t</sup>],b<sub>a</sub>)
  *  Identify a memory cell : c<sup>t</sup>=a<sup>t</sup>
  * Define gate: Gamma of u as a sigmoid function (close to 0 to 1). Think of this as a gated fence.
  * Gate will decide whether to update the memory cell or not between the layers
  * The c vector, C~ vector and the gamma vector will all have the same dimensions
  * Simplified GRU
     ![Reference](https://github.com/geethaRam/machinelearning/blob/master/gru.png "Logo Title Text 1")
  
  * Full GRU   
     ![Reference](https://github.com/geethaRam/machinelearning/blob/master/gru1.png "Logo Title Text 1")
  
  ## LSTM - Long Short term memory
  * Powerful than GRU
  * But GRU is faster since there are only two gates and hence hte number of cmputations are less - which means it can scale       well to many number of layers.
  * GRU vs LSTM equation
     ![Reference](https://github.com/geethaRam/machinelearning/blob/master/lstm.png "Logo Title Text 1")
     
 ## Bi-directional RNN
  * getting information from future.
  * So far - we have only seen RNN models that take information from previous words/layers. 
  * In cases like below, we need to take into account the words from the future as well
  *  Sequence1 : He said, "I like Teddy bears"
     Sequence2: He said, "I like Teddy Roosevelt since he is the best president!"
  * The first few words would not tell whether they are talking about `teddy bears` or `presidents`
  * A Bi-directional stage is added to each stage as part of the forward prop itself.
  * Any NLP problem usually needs Bi-directional RNN with LSTM blocks for Forward Prop
  * Disadvantage: Need the entire sequence of data. So, cannot do any real -time speech recognition type problems.
  * BRNN
     ![Reference](https://github.com/geethaRam/machinelearning/blob/master/brnn.png "Logo Title Text 1")
 
 ## Deep RNN
 * The RNN cannot have many deep layers. Most of them would not have more than 3 deep layers since RNNs are already much           temporal in sense. 
   ![Reference](https://github.com/geethaRam/machinelearning/blob/master/deep_rnn.png "Logo Title Text 1")
     
  
 
 
 
 
 





