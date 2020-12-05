"""Model classes and model utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy 

from utils import cuda, load_cached_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.elmo import Elmo, batch_to_ids



def _sort_batch_by_length(tensor, sequence_lengths):
    """
    Sorts input sequences by lengths. This is required by Pytorch
    `pack_padded_sequence`. Note: `pack_padded_sequence` has an option to
    sort sequences internally, but we do it by ourselves.

    Args:
        tensor: Input tensor to RNN [batch_size, len, dim].
        sequence_lengths: Lengths of input sequences.

    Returns:
        sorted_tensor: Sorted input tensor ready for RNN [batch_size, len, dim].
        sorted_sequence_lengths: Sorted lengths.
        restoration_indices: Indices to recover the original order.
    """
    # Sort sequence lengths
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    # Sort sequences
    sorted_tensor = tensor.index_select(0, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


class AlignedAttention(nn.Module):
    """
    This module returns attention scores over question sequences. Details can be
    found in these papers:
        - Aligned question embedding (Chen et al. 2017):
             https://arxiv.org/pdf/1704.00051.pdf
        - Context2Query (Seo et al. 2017):
             https://arxiv.org/pdf/1611.01603.pdf

    Args:
        p_dim: Int. Passage vector dimension.

    Inputs:
        p: Passage tensor (float), [batch_size, p_len, p_dim].
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over question sequences, [batch_size, p_len, q_len].
    """
    def __init__(self, p_dim):
        super().__init__()
        self.linear = nn.Linear(p_dim, p_dim)
        self.relu = nn.ReLU()

    def forward(self, p, q, q_mask):
        # Compute scores
        p_key = self.relu(self.linear(p))  # [batch_size, p_len, p_dim]
        q_key = self.relu(self.linear(q))  # [batch_size, q_len, p_dim]
        scores = p_key.bmm(q_key.transpose(2, 1))  # [batch_size, p_len, q_len]
        # Stack question mask p_len times
        q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along question length
        return F.softmax(scores, 2)  # [batch_size, p_len, q_len]


class SpanAttention(nn.Module):
    """
    This module returns attention scores over sequence length.

    Args:
        q_dim: Int. Passage vector dimension.

    Inputs:
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over sequence length, [batch_size, len].
    """
    def __init__(self, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, 1)

    def forward(self, q, q_mask):
        # Compute scores
        q_scores = self.linear(q).squeeze(2)  # [batch_size, len]
        # Assign -inf to pad tokens
        q_scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along sequence length
        return F.softmax(q_scores, 1)  # [batch_size, len]


class BilinearOutput(nn.Module):
    """
    This module returns logits over the input sequence.

    Args:
        p_dim: Int. Passage hidden dimension.
        q_dim: Int. Question hidden dimension.

    Inputs:
        p: Passage hidden tensor (float), [batch_size, p_len, p_dim].
        q: Question vector tensor (float), [batch_size, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Logits over the input sequence, [batch_size, p_len].
    """
    def __init__(self, p_dim, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, p_dim)


    def binsearch(self, list, toFind):
        low = 0
        n = len(list)
        high = n-1

        while low + 1 < high:
            mid = low + ((high - low)//2)
            if list[mid] == toFind:
                return mid
            elif list[mid] < toFind:
                low = mid
            else:
                high = mid

        if list[low] == toFind:
            return low
        else:
            return high


    def forward(self, p, q, p_mask, passagesNer, questionsNer, passagesMaps, questionsMaps, rawPassages, rawQuestions, trueCasePassages, trueCaseQuestions):
        # Compute bilinear scores
        q_key = self.linear(q).unsqueeze(2)  # [batch_size, p_dim, 1]
        p_scores = torch.bmm(p, q_key).squeeze(2)  # [batch_size, p_len]
        # Assign -inf to pad tokens
        p_scores.data.masked_fill_(p_mask.data, -float('inf'))
        #print ("pmaskdata: " + str(p_mask.data))
        # create set of all named entities in the question
        questionNerSets = list()
        for doc in questionsNer:
            questionNerSet = set()
            for ent in doc.ents:
                questionNerSet.add(ent.text.lower())
            questionNerSets.append(questionNerSet)

        for i in range(len(passagesNer)):
            
            who = "who" in rawQuestions[i]
            who_entities = ["PERSON", "ORG", "GPE", "NORP"]

            when = "when" in rawQuestions[i]
            when_entities = ["DATE", "TIME"]

            quantity = False
            for x in range(0, len(rawQuestions[i])):
                if (rawQuestions[i][x] == "how" and (rawQuestions[i][x + 1] == "much" or rawQuestions[i][x + 1] == "many")):
                    quantity = True
            quantity_entities = ["MONEY", "QUANTITY", "PERCENT", "CARDINAL"]

            questionEntities = questionNerSets[i]
            #print ("questionEntities: " + str(questionEntities))            

            doc = passagesNer[i]
            for ent in doc.ents:
                sc = ent.start_char
                ec = ent.end_char

                wordIndex = self.binsearch(passagesMaps[i], sc)

                curChar = sc

                actualWord = trueCasePassages[i][wordIndex].lower()

                while curChar + len(actualWord) <= ec:

                    actualWord = trueCasePassages[i][wordIndex].lower()

                    '''
                    if actualWord not in ent.text.lower():
                        print ("rawPassage: " + str(rawPassages[i]))
                        print ("rawQuestion: " + str(rawQuestions[i]))
                        print ("questionEntities: " + str(questionEntities))
                        print ("word we were supposed to find: " + str(ent.text.lower()))
                        print ("actualWordWithOurghettobinsearchshit: " + str(actualWord))
                        #exit()
                        break
                    
                    #print("p_scores[i]: " + str(len(p_scores[i])))
                    #print("trueCasePassages[i]: " + str(len(trueCasePassages[i])))
                    '''

                    # '''
                    #if this named entity in the passage is in the question, then do pscores[i]++ lmao
                    if actualWord in questionEntities:
                        #print ("actualWordMatched: " + str(actualWord))
                        #print ("before: " + str(p_scores[i][wordIndex]))
                        p_scores[i][wordIndex] -= 1 #idk
                        
                        if who and ent.label_ in who_entities:
                            p_scores[i][wordIndex] -= 0.5

                        if when and ent.label_ in when_entities:
                            p_scores[i][wordIndex] -= 0.5

                        if quantity and ent.label_ in quantity_entities:
                            p_scores[i][wordIndex] -= 0.5
                    # '''

                    wordIndex += 1
                    curChar += len(actualWord) + 1 # 1 for the space pls
                    
            #exit()

        return p_scores  # [batch_size, p_len]


class BaselineReader(nn.Module):
    """
    Baseline QA Model
    [Architecture]
        0) Inputs: passages and questions
        1) Embedding Layer: converts words to vectors
        2) Context2Query: computes weighted sum of question embeddings for
               each position in passage.
        3) Passage Encoder: LSTM or GRU.
        4) Question Encoder: LSTM or GRU.
        5) Question Attentive Sum: computes weighted sum of question hidden.
        6) Start Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.
        7) End Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.

    Args:
        args: `argparse` object.

    Inputs:
        batch: a dictionary containing batched tensors.
            {
                'passages': LongTensor [batch_size, p_len],
                'questions': LongTensor [batch_size, q_len],
                'start_positions': Not used in `forward`,
                'end_positions': Not used in `forward`,
            }

    Returns:
        Logits for start positions and logits for end positions.
        Tuple: ([batch_size, p_len], [batch_size, p_len])
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.pad_token_id = args.pad_token_id
        self.spacy = spacy.load('en_core_web_sm') 

        # Initialize embedding layer (1)
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)

        # Initialize Context2Query (2)
        self.aligned_att = AlignedAttention(args.embedding_dim)

        rnn_cell = nn.LSTM if args.rnn_cell_type == 'lstm' else nn.GRU

        # Initialize passage encoder (3)
        self.passage_rnn = rnn_cell(
            args.embedding_dim * 2,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        # Initialize question encoder (4)
        self.question_rnn = rnn_cell(
            args.embedding_dim,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(self.args.dropout)

        # Adjust hidden dimension if bidirectional RNNs are used
        _hidden_dim = (
            args.hidden_dim * 2 if args.bidirectional
            else args.hidden_dim
        )

        # Initialize attention layer for question attentive sum (5)
        self.question_att = SpanAttention(_hidden_dim)

        # Initialize bilinear layer for start positions (6)
        self.start_output = BilinearOutput(_hidden_dim, _hidden_dim)

        # Initialize bilinear layer for end positions (7)
        self.end_output = BilinearOutput(_hidden_dim, _hidden_dim)

        

    def load_pretrained_embeddings(self, vocabulary, path, sentences):
        """
        Loads GloVe vectors and initializes the embedding matrix.

        Args:
            vocabulary: `Vocabulary` object.
            path: Embedding path, e.g. "glove/glove.6B.300d.txt".
        """

        self.vocabulary = vocabulary


        '''options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Note the "1", since we want only 1 output representation for each token.
        elmo = Elmo(options_file, weight_file, 1, dropout=0)
        print ("made it here the prequel")
        sentences = sentences[0:5]
        print (sentences)
        character_ids = batch_to_ids(sentences)
        print ("made it here")
        print ("charids: " + str(character_ids))
        embeddings = elmo(character_ids)'''
        

        #return len(vocabulary)

        embedding_map = load_cached_embeddings(path)
        



        # Create embedding matrix. By default, embeddings are randomly
        # initialized from Uniform(-0.1, 0.1).
        embeddings = torch.zeros(
            (len(vocabulary), self.args.embedding_dim)
        ).uniform_(-0.1, 0.1)

        # Initialize pre-trained embeddings.
        num_pretrained = 0
        for (i, word) in enumerate(vocabulary.words):
            if word in embedding_map:
                embeddings[i] = torch.tensor(embedding_map[word])
                num_pretrained += 1

        # Place embedding matrix on GPU.
        self.embedding.weight.data = cuda(self.args, embeddings)

        return num_pretrained
        #return 0

        
    def sorted_rnn(self, sequences, sequence_lengths, rnn):
        """
        Sorts and packs inputs, then feeds them into RNN.

        Args:
            sequences: Input sequences, [batch_size, len, dim].
            sequence_lengths: Lengths for each sequence, [batch_size].
            rnn: Registered LSTM or GRU.

        Returns:
            All hidden states, [batch_size, len, hid].
        """
        # Sort input sequences
        sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
            sequences, sequence_lengths
        )
        # Pack input sequences
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, _ = rnn(packed_sequence_input, None)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def forward(self, batch):
        # Obtain masks and lengths for passage and question.
        
        '''options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Note the "1", since we want only 1 output representation for each token.
        elmo = Elmo(options_file, weight_file, 1, dropout=0)


        

        # 1) Embedding Layer: Embed the passage and question.
        #passage_embeddings = self.embedding(batch['passages'])  # [batch_size, p_len, p_dim]
        oldQuestionEmbeddings = self.embedding(batch['questions'])  # [batch_size, q_len, q_dim]

        oldPassageEmbeddings = self.embedding(batch['passages'])
        print ("$$$$$$$$$$$$$:" + str(type(oldPassageEmbeddings)))
        print ("old one  shape: " + str(oldPassageEmbeddings.shape))
        print ("old question embeddings shape: " + str(oldQuestionEmbeddings.shape))
        #passagesList = batch['rawPassages']

        #print ("passagesList: " + str(passagesList))


        passage_character_ids = batch_to_ids(batch["rawPassages"])
        passage_embeddings = elmo(passage_character_ids)["elmo_representations"][0]

        print ("pre *********: " + str(type(passage_embeddings)))
        print ("*************:" +str(passage_embeddings))
        print ("new oneshape: " + str(passage_embeddings.shape))

        question_character_ids = batch_to_ids(batch['rawQuestions'])
        question_embeddings = elmo(question_character_ids)["elmo_representations"][0]'''


        # 2) Context2Query: Compute weighted sum of question embeddings for
        #        each passage word and concatenate with passage embeddings.

        passage_mask = (batch['passages'] != self.pad_token_id)  # [batch_size, p_len]
        question_mask = (batch['questions'] != self.pad_token_id)  # [batch_size, q_len]
        passage_lengths = passage_mask.long().sum(-1)  # [batch_size]
        question_lengths = question_mask.long().sum(-1)  # [batch_size]
        
        passage_embeddings = self.embedding(batch['passages'])
        question_embeddings = self.embedding(batch['questions'])

        aligned_scores = self.aligned_att(
            passage_embeddings, question_embeddings, ~question_mask
        )  # [batch_size, p_len, q_len]
        aligned_embeddings = aligned_scores.bmm(question_embeddings)  # [batch_size, p_len, q_dim]
        passage_embeddings = cuda(
            self.args,
            torch.cat((passage_embeddings, aligned_embeddings), 2),
        )  # [batch_size, p_len, p_dim + q_dim]

        # 3) Passage Encoder
        passage_hidden = self.sorted_rnn(
            passage_embeddings, passage_lengths, self.passage_rnn
        )  # [batch_size, p_len, p_hid]
        passage_hidden = self.dropout(passage_hidden)  # [batch_size, p_len, p_hid]

        # 4) Question Encoder: Encode question embeddings.
        question_hidden = self.sorted_rnn(
            question_embeddings, question_lengths, self.question_rnn
        )  # [batch_size, q_len, q_hid]

        # 5) Question Attentive Sum: Compute weighted sum of question hidden
        #        vectors.
        question_scores = self.question_att(question_hidden, ~question_mask)
        question_vector = question_scores.unsqueeze(1).bmm(question_hidden).squeeze(1)
        question_vector = self.dropout(question_vector)  # [batch_size, q_hid]

        rawPassages = batch['rawPassages']
        rawQuestions = batch['rawQuestions']

        trueCasePassages = batch['trueCasePassages']
        trueCaseQuestions = batch['trueCaseQuestions']

        passagesNer = list()
        passagesMaps = list()
        for passage in trueCasePassages:
            startOfEachWord = list() # list of tuples of (char index (start), word index)

            joinedString = " ".join(passage)

            newWord = True
            wordCounter = 0
            for i in range(len(joinedString)):
                if newWord:
                    startOfEachWord.append(i)
                    wordCounter += 1
                    newWord = False
                if joinedString[i] == " ":
                    newWord = True


            pdoc = self.spacy(joinedString)
            passagesNer.append(pdoc)
            passagesMaps.append(startOfEachWord)

        questionsNer = list()
        questionsMaps = list()
        for question in trueCaseQuestions:
            startOfEachWord = list()
            joinedString = " ".join(question)
            newWord = True
            wordCounter = 0
            for i in range(len(joinedString)):
                if newWord:
                    startOfEachWord.append(i)
                    wordCounter += 1
                    newWord = False
                if joinedString[i] == " ":
                    newWord = True

            qdoc = self.spacy(joinedString)
            questionsNer.append(qdoc)
            questionsMaps.append(startOfEachWord)





        # 6) Start Position Pointer: Compute logits for start positions
        start_logits = self.start_output(
            passage_hidden, question_vector, ~passage_mask, passagesNer, questionsNer, passagesMaps, questionsMaps, rawPassages, rawQuestions, trueCasePassages, trueCaseQuestions
        )  # [batch_size, p_len]

        # 7) End Position Pointer: Compute logits for end positions
        end_logits = self.end_output(
            passage_hidden, question_vector, ~passage_mask, passagesNer, questionsNer, passagesMaps, questionsMaps, rawPassages, rawQuestions, trueCasePassages, trueCaseQuestions
        )  # [batch_size, p_len]

        return start_logits, end_logits  # [batch_size, p_len], [batch_size, p_len]
