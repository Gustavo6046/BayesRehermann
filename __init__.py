"""
BRCCS - Bayes-Rehermann Conversational Classification System

Inspired by Markov chains and using naive Bayes classifiers,
this system is a conversational response system that works
by classifying every input sentence I's feature set, with a
response word index T, into the predicted output word at the
index T. Once T is larger than the predicted output's length,
it should return None.

This is a technology developed by Gustavo6046, and as such,
falls under the clauses of the MIT License for source code.

(c)2018 Gustavo R. Rehermann.
"""
import random
import nltk
import pandas
import sqlite3

from nltk.stem.porter import *
from threading import Thread


stemmer = PorterStemmer()


class BayesRehermann(object):
    """
    The Bayes-Rehermann classification system.
    
    An experimental conversational design, for a generative
    chatbot system. The name comes from two facts:
    
    * the model was designed by Gustavo R. Reherman
      (Gustavo6046);
    
    * the model utilizes mainly naive Bayes classification to
      achieve what it needs.
    """

    def __init__(self, database=None):
        """
        Initializes the Bayes-Rehermann classification system.
        database should be the filename of the sqlite database
        to use to keep and retrieve snapshots.
        """
    
        self.data = []
        self.classifiers = {}
        self.history = {}
        self.conversation_ids = {}
        self.snapshots = {}
        self.database = database
        
        if database is not None:
            c = self.conn().cursor()
            
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='SnapIndex';")
            
            if len(c.fetchall()) < 1:
                c.execute("CREATE TABLE SnapIndex (name text, sindex int);")
                
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='History';")
            
            if len(c.fetchall()) < 1:
                c.execute("CREATE TABLE History (speaker text, sentence text);")
                self.conn().commit()
            
            c.execute("SELECT * FROM SnapIndex;")
            
            for name, index in c.fetchall():
                c.execute("SELECT * FROM Snapshot_{};".format(index))
                contexts = []
                
                for cind, sentence in c.fetchall():
                    while cind >= len(contexts):
                        contexts.append([])
                    
                    contexts[cind].append(sentence)
                    
                self.add_snapshot(name, contexts, message_handler=print, commit=False, use_threads=False)
                
            c.execute("SELECT * FROM History;")
            
            for speaker, sentence in c.fetchall():
                if speaker not in self.history:
                    self.history[speaker] = []
                    
                self.history[speaker].append(sentence)
        
    def conn(self):
        if self.database is not None:
            return sqlite3.connect(self.database)
            
        else:
            return None
        
    def add_snapshot(self, name, data, *args, **kwargs):
        """
        Adds and trains a snapshot to the system.
        
        This function will also train a classifier, so
        you can quickly retrieve input from the snapshot of same name.
        """
    
        old = self.data
        self.data = data
        res = self.create_snapshot(name, *args, **kwargs)
        self.data = old
        
        return res

    def sentence_data(self, sent, history, use_context=True, history_limit=10, **kwargs):
        """
        Returns the feature set used in the classifier. Feel free to
        replace in subclasses :)
        """
    
        tokens = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(tokens)
        
        data = kwargs
        
        data['total chars'] = len(sent)
        data['total words'] = len(sent.split(' '))
        data['total tokens'] = len(tokens)
            
        for i, (word, tag) in enumerate(tags):
            def sub_data(name, value):
                data["{} #{}".format(name, i)] = value
                data["{} #-{}".format(name, len(tags) - i)] = value
            
            sub_data('tag', tag)
            sub_data('token', word)
            sub_data('pos', (word, tag))
            sub_data('token chars', len(word))
            sub_data('tag stem', tag[:2])
            sub_data('tag branch', tag[2:])
            sub_data('token stem', stemmer.stem(word))
            sub_data('first letter', word[0])
            sub_data('first letter', word[-1])
            
        if use_context:
            for i, h in enumerate(history[-i + 1 + history_limit:][::-1]):
                for k, v in self.sentence_data(h, history[i + 1:], use_context=False).items():
                    data['-{} {}'.format(i, k)] = v
            
        return data
        
    def create_snapshot(self, key, clear_data=True, message_handler=print, commit=True, use_threads=True):
        """
        Creates a snapshot using the current sentence data buffer.
        """
    
        # Check if the snapshot already exists. It should be a grow-only, no-replacement database.
        if key in self.snapshots:
            if message_handler is not None:
                message_handler("The snapshot '{}' already exists!".format(key))
            
            return False
            
        # Create a snapshot.
        self.snapshots[key] = self.data
        
        # Trains a classifier from the snapshot.
        #
        # This very classifier is what the snapshot system exists;
        # to avoid having to retrain a classifier at runtime everytime
        # we want to get some output from the BRCCS.
        train_data = []
            
        # Commits the new snapshot to the sqlite database, if asked to.
        if self.database is not None and commit:
            c = self.conn().cursor()
            c.execute("CREATE TABLE Snapshot_{} (context int, sentence text);".format(len(self.snapshots) - 1))
            c.execute("INSERT INTO SnapIndex VALUES (?, ?);", (key, len(self.snapshots) - 1))
            
            for i, context in enumerate(self.snapshots[key]):
                for sentence in context:
                    c.execute("INSERT INTO Snapshot_{} VALUES (?, ?);".format(len(self.snapshots) - 1), (i, sentence))
            
            self.conn().commit()
             
        if message_handler is not None:
            message_handler("Constructing training data for snapshot '{}'...".format(key))
             
        for context in self.data:
            for i, sentence in enumerate(context[:-1]):
                for wi, word in list(enumerate(context[i + 1].split(' ') + [False] * 50)):
                    train_data.append((self.sentence_data(sentence, context[:i], response_index=wi), word))
            
        def train():
            if message_handler is not None:
                message_handler("Training snapshot '{}'...".format(key))
            
            if len(train_data) > 0:
                # print(train_data[0])
                self.classifiers[key] = nltk.NaiveBayesClassifier.train(train_data)
                
            else:
                raise ValueError("No training data from snapshot '{}'!".format(key))
            
            if message_handler is not None:
                message_handler("Snapshot '{}' trained successfully!".format(key))
                
            if clear_data:
                self.data = []
                self.conversation_ids = {}
                
        if use_threads:
            Thread(target=train).start()
            
        else:
            train()
        
        return True
        
    def add_conversation(self, conversation, id=None):
        """
        Adds a list of sentences, in a conversational format, to the current
        data buffer. A sequence of add_conversation calls, followed by create_snapshot,
        will create a snapshot and a classifier for this conversation. Alternatvely, you can
        use a list of conversations and add_snapshot.
        
        If you want to grow the conversation later, provide an ID, so you can use the
        grow_conversation method later on.
        """
    
        if id is not None:
            self.conversation_ids[id] = len(self.data)

        self.data.append(conversation)
        
    def restore_snapshot(self, snapshot):
        """
        Restore a snapshot, extending it into the conversational buffer, usually to further develop it into
        another snapshot, or include it as part of another snapshot (which technically is the same).
        """
        res = snapshot in self.snapshots
        
        if res:
            self.data.extend(self.snapshots[snapshot])
            
        return res
        
    def grow_conversation(self, id, conversation):
        """
        Extend a conversation, if you provided an ID in the add_conversation call.
        Useful for dynamic environments, like IRC.
        """
        if id not in self.conversation_ids:
            self.add_conversation(conversation, id)
            
        else:
            self.data[self.conversation_ids[id]].extend(conversation)
        
    def reset_id(self, id):
        """
        Resets a conversation ID, so the next time you use it,
        it'll point to a new conversation. Useful for dynamic environments,
        like IRC, where conversations may be split by longer time periods;
        for you'll want to reset the conversation ID (server name + channel
         ame) at every split.
        """
        b = id in self.conversation_ids
        
        if b:
            self.conversation_ids.pop(id)
            
        return b
        
    def respond(self, snapshot, sentence, speaker=None, use_history=True, commit_history=True, limit=1000, recursion_limit=5):
        """
        Returns the response to the given sentence, predicted by the classifier of the
        corresponding snapshot.
        
        The recursion limit exists because naive Bayes classifiers weren't really made for
        this, so after a certain index, they would just keep outputting the same word. A
        check was implemented to detect and avoid those.
        """
    
        if speaker is None or not use_history:
            history = []
            
        else:
            history = self.history.get(speaker, [])
        
        c = self.classifiers[snapshot]
        response = []
        
        i = 0
        
        last = None
        recurse = 0
        
        while True:
            word = c.classify(self.sentence_data(sentence, history, response_index=i))
            
            if word is False:
                break
                
            if word == last:
                recurse += 1
                
            else:
                recurse == 0
                
            if recurse > recursion_limit:
                response = response[:-recurse + 1]
                break
                
            response.append(word)
            i += 1
            
            if len(response) >= limit:
                break
                
            last = word
            recursion_limit = min(recursion_limit, limit - len(response))
        
        if use_history and speaker is not None:
            self.grow_conversation("__RESPONSE_HISTORY:{}__".format(speaker), [sentence, ' '.join(response)])
        
            if speaker not in self.history:
                self.history[speaker] = []
                
            self.history[speaker].append(sentence)
            self.history[speaker].append(' '.join(response))
            
            if commit_history:
                c = self.conn().cursor()
                
                c.execute("INSERT INTO History VALUES (?, ?);", (speaker, sentence))
                c.execute("INSERT INTO History VALUES (?, ?);", (speaker, ' '.join(response)))
        
        return ' '.join(response)