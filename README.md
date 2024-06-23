# MIT_6.86x
# Machine Learning with Python-From Linear Models to Deep Learning
# Unit 1
# Lecture 1. Introduction to Machine Learning
 **2. Objectives**
  Introduction to Machine Learning
  
>Understand the goal of machine learning from a movie recommender example
>Understand elements of supervised learning, and the difference between the training set and the test set
>Understand the difference of classification and regression - two representative kinds of supervised learning

 **3. What is Machine Learning?**

Machine learning as a discipline aims to design, understand, and apply computer programs that learn from experience (i.e. data) for the purpose of modelling, prediction, and control. We will start with prediction as a core machine learning task.

There are many types of predictions that we can make. We can predict outcomes of events that occur in the future such as the market, weather tomorrow, the next word a text message user will type, or anticipate pedestrian behavior in self driving vehicles, and so on.

We can also try to predict properties that we do not yet know. For example, properties of materials such as whether a chemical is soluble in water, what the object is in an image, what an English sentence translates to in Hindi, whether a product review carries positive sentiment, and so on.

 **4. Introduction to Supervised Learning**

Common to all these “prediction problems" mentioned on the previously is that it is very hard to write down a solution in terms of rules or code directly, and far easier to provide examples of correct behavior. For example, how would you encode rules for translation, or image classification? It is much easier to provide large numbers of translated sentences, or examples of what the objects are on a large set of images. The ability to learn the solution from examples is what has made machine learning so popular and pervasive.

We will start with supervised learning in this course. In supervised learning, we are given an example (e.g. an image) along with a target (e.g. what object is in the image), and the goal of the machine learning algorithm is to find out how to produce the target from the example.

More specifically, in supervised learning, we hypothesize a collection of functions (or mappings) parametrized by a parameter, from the examples (e.g. the images) to the targets (e.g. the objects in the images). The machine learning algorithm then automates the process of finding the parameter of the function that fits with the example-target pairs the best.

 **5. A Concrete Example of a Supervised Learning Task**

   **Feature Vector Demystified 1**

We have a movie recommending system that reads description of each movie and determines some important characteristics of the movie. In particular, it examines whether each of the criterion below is true for that movie:

1.Is it a comedy movie?
2.Is it an action movie?
3.Was the movie directed by Spielberg?
4.Do dinosaurs appear in the movie?
5.Is it a Disney film?

For example, when the recommending system reads descriptions of "Jurassic Park", the answers for the five questions above will be "no, yes, yes, yes, no." On the other hand if the recommending system reads descriptions of "High School Musical", the answers will be "no, no, no, no, yes"

The system converts "yes" into 1, "no" into 0, and makes a feature vector **X** for each movie. So **X_{Jurrasic Park} ** will be \big [0,1,1,1,0\big ], while X_{High School Musical}will be X_{High School Musical} \big [0,0,0,0,1\big ]

Question 1: Now we have a comedy movie that is not an action movie, that was not directed by Spielberg, that does not have dinosaurs in it, but was produced by Disney. What is this movie's feature vector?
Ans = \big [1,0,0,0,1\big ]

   **Feature Vector Demystified 2**

Question 2: What is the dimension of the feature vector of this movie? Ans = 5

  **Training Set vs Test Set 1**

The ultimate goal of our recommending system is to predict whether John will like this movie. Now suppose our movie recommending system knows whether John likes or dislikes the following movies:

comedy	action	Spielberg	Dinosaur Appearance	Disney	Liked by John?
movie 1	0	1	0	0	1	1
movie 2	1	1	1	0	0	-1
movie 3	0	1	0	1	1	1
movie 4	1	1	0	1	0	1
(Like is denoted as **1** and dislike as **-1** in the above table) On the other hand, the movie recommender does not know whether John likes the following movies when building the model, but will know them after the model is built:

comedy	action	Spielberg	Dinosaur Appearance	Disney	Liked by John?
movie 5	1	0	0	0	0	Don't know yet
movie 6	0	0	0	0	1	Don't know yet
movie 7	0	0	0	1	1	Don't know yet
Assume that, when John evaluates movies, he only does so based on the five criteria.

Question 1: What is the label of movie 1, based on the fact that John likes the movie? Ans = 1

  **Training Set vs Test Set 2**

Question 2: What movies are in the training set? Select all those apply. Ans = movie 1,2,3,4

  **Training Set vs Test Set 3**

Question 3: What movies are in the test set? Select all those apply. Ans = movie 5,6,7

**6. Introduction to Classifiers: Let's bring in some geometry!**

Training data can be graphically depicted on a (hyper)plane. Classifiers are mappings that take feature vectors as input and produce labels as output. A common kind of classifier is the linear classifier, which linearly divides space(the (hyper)plane where training data lies) into two. Given a point **X** in the space, the classifier **h** outputs **h(x) = 1**  or **h(x) = 1** , depending on where the point **X** exists in among the two linearly divided spaces.

  **Linear Classifier**
We have a linear classifier **h** that takes in any point on a two-dimensional space. The linear classifier **h** divides the two-dimensional space into two, such that on one side **h(x) = +1** and on the other side **h(x) = -1**, as depicted below.

![image](https://github.com/iamkushvanth/MIT_6.86x/assets/160105601/589273b1-0ca0-4b15-ace3-a80399709966)

For x = (10,10) , would h(x) be -1 or +1 ? Ans = +1

  **Training Error**
Suppose a classifier correctly classifies 5 points in the training set and 1 points in the test set. Suppose it incorrectly classifies 5 points in the training set and 2 points in the test set. What is the training error? Is it better than chance? Ans = 0.5, equal to chance

   ** Hypothesis Space**
What is the meaning of the "hypothesis space"? Ans = the set of possible classifiers

 **7. Different Kinds of Supervised Learning: classification vs regression**   

 **Classification vs regression**
    
   Classification maps feature vectors to categories. The number of categories need not be two - they can be as many as needed. Regression maps feature vectors to real numbers. There are other kinds of supervised learning as well.

For a more thorough statistical background on classification and regression, please check out the following links. [Classification Regression](https://en.wikipedia.org/wiki/Regression_analysis)

  **Classification or Regression? 1**

  Question 1: We want to come up with a classifier that classifies each news article into one of the following categories: politics, sports, entertainment. Is this a classification problem or a regression problem? Ans = classification

  Question 2: We want to estimate the price of bitcoin after 30 days. Is this a classification problem or a regression problem? Ans= regression

  Choose the type of learning that best corresponds to each of the following statements.

1)Labelled training and test examples Ans = supervised learning

2)Using knowledge from one task to solve another task Ans = transfer learning

3)Learning to navigate a robot Ans = reinforcement learning

4)Deciding which examples are needed to learn Ans =  active learning

5)Data with no annotation Ans = unsupervised learning

6)Training and test examples with limited annotation Ans = semi-supervised learning

# Lecture 2. Linear Classifier and Perceptron  

**1. Objectives**

Linear Classifiers and Perceptron Algorithm

At the end of this lecture, you will be able to

> understand the concepts of Feature vectors and labels, Training set and Test set, Classifier, Training error, Test error, and the Set of classifiers

> derive the mathematical presentation of linear classifiers

> understand the intuitive and formal definition of linear separation

> use the perceptron algorithm with and without offset

**2. Review of Basic Concepts**

**Concept Review Problem: car accident prediction 1**
In this problem, we will put ourselves in the shoes of a car insurance company. Our goal is to find out whether customers were involved in an accident on July 4th, 1998.
For 8 customers, we know the following information:

1. number of accidents the customer made in the past.
2. number of miles the customer has driven.
3. the customer's age

Also, for 5 of the customers, we know whether each of them was involved in an accident on July 4th, 1998.
If we want to learn a model in a supervised way, what is n, the number of training examples? n= 5

**Concept Review Problem: car accident prediction 2**

The insurance company recorded relevant information for all 8 customers, as illustrated in the table below.
            number of past accidents |	miles customer drove so far |	customer's age	
customer 1		         0             |          2710.9	            |     21
customer 2		         2             |          13209.2	            |     40
customer 3		         1             |          89001.4             |     32
customer 4		         3             |          12381.1	            |     18
customer 5	           0             |          1893.5	            |     24
customer 6	           2	           |          32493.5	            |     24
customer 7	           1 	           |          5443.5	            |     30
customer 8	           0	           |          4493.5	            |     28
What is the dimension of each feature vector? d= 3

**Concept Review Problem: car accident prediction 3**

How many feature vectors are there in the above table? Ans = 8

**Concept Review Problem: Classifier and Training Error 1**

Assume we have training data and a classifier like the following: (where h(x) denotes the value outputted by the classifier with the data point as input)
 h(  x , y )
data 1	1	1
data 2	-1	1
data 3	1	1
data 4	1	-1
data 5	-1	-1

What is the training error? E_n(h) = 0.6

**Concept Review Problem: Classifier and Training Error 2**

Now let's examine the training error E_n(h)  in a general sense. E_n(h)  is a function of: (choose all those apply)
Ans = n, the number of training data
      h , the classifier

      
**3. Linear Classifiers Mathematically Revisited**

**Linear Classifiers Mathematically Revisited**

**Inner product and Orthogonal vectors**
What is the inner product of [0,1,1]  and [1,1,1] ? Ans = 2

**Linear Classifier Practice**
We saw in the lecture above that for a linear classifier h , h(x;\theta ) = sign(\theta \cdot x), i.e. the sign of the dot product of \theta and x . Now consider \theta which is given by
\theta = (1,-1)  (4.1)
Which of the following points would be classified as positive by ? Please choose all correct answers. Ans = (1,0)

**Offset Added**
Again, we have a linear classifier with \theta given by
\theta = (1,-1)  (4.2)
and the offset, \theta_0 given by \theta =-1 Now which of the following points would be classified as positive by \theta ? Please choose all correct answers. Ans = (1,-1)






