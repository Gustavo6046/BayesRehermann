# BRCCS - Bayes-Rehermann Conversational Classification System

Inspired by Markov chains and using naive Bayes classifiers,
this system is a conversational response system for generative
chatbots.

The name comes from two facts:

* Bayes because the model originally used mainly naive Bayes
  classification to achieve performant training and response.
  (now it uses decision trees).

* Rehermann because the model was designed by Gustavo R.
  Reherman (Gustavo6046)!

It works by classifying every input sentence I's feature set,
with a response word index T, into the predicted output word
at the index T. Once T is larger than the predicted output's
length, it should return None.
  
This is a technology developed by Gustavo6046, and as such,
falls under the clauses of the MIT License for source code.

©2018 Gustavo R. Rehermann.