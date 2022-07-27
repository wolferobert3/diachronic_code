import torch
import numpy as np
import math
from scipy.stats import norm, pearsonr

def bert_loss(model, tokenizer, masked_sentence, target_word):

  inputs = tokenizer(masked_sentence, return_tensors="pt")

  labels = tokenizer(masked_sentence.replace('[MASK]', target_word), return_tensors="pt")["input_ids"]
  labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

  outputs = model(**inputs, labels=labels)
  loss = round(outputs.loss.item(), 2)

  return loss

def bert_top_words(model, tokenizer, masked_sentence, num_words):

    inputs = tokenizer(masked_sentence, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_ids = (-logits[0, mask_token_index]).argsort()[:num_words]
    predicted_tokens = [tokenizer.decode(id) for id in predicted_ids]

    #predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    #prediction = tokenizer.decode(predicted_token_id)

    return predicted_tokens

def gpt2_perplexity(model, tokenizer, sentence):

      input_ids = torch.tensor(tokenizer.encode(sentence))

      with torch.no_grad():
          outputs = model(input_ids, labels=input_ids)

      loss = outputs[0]
      perplexity = math.exp(loss)

      return perplexity

def gpt2_loss(model, tokenizer, sentence):

      input_ids = torch.tensor(tokenizer.encode(sentence))

      with torch.no_grad():
          outputs = model(input_ids, labels=input_ids)

      loss = outputs[0]
      perplexity = math.exp(loss)

      return perplexity

def gpt2_predictions(model, tokenizer, sentence, num_words):

    input_ids = torch.tensor(tokenizer.encode(sentence))

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs[1]
    predicted_ids = (-logits[0, -1]).argsort()[:num_words]

    predicted_tokens = [tokenizer.decode(id) for id in predicted_ids]

    return predicted_tokens

def most_similar_words(target_emb, embedding_matrix, word_list, num_words):

    scaled_target = target_emb / np.linalg.norm(target_emb)
    scaled_embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis = 1)

    similarities = scaled_embedding_matrix @ scaled_target.T
    top_similarities = (-similarities).argsort()[:num_words]
    
    words = [word_list[i] for i in top_similarities]
    cosine_similarities = [similarities[i] for i in top_similarities]

    return words, cosine_similarities

def sc_weat(w, A, B, permutations):

    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=1, keepdims=True)

    A_associations = A_normed @ w_normed.T
    B_associations = B_normed @ w_normed.T
    joint_associations = np.concatenate((A_associations, B_associations), axis=1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations, ddof=1)

    midpoint = A.shape[0]
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in permutations])
    sample_associations = np.mean(sample_distribution[:, :midpoint]) - np.mean(sample_distribution[:, midpoint:])
    p_value = 1 - norm.cdf(test_statistic, np.mean(sample_associations), np.std(sample_associations, ddof=1))

    return effect_size, p_value

def sc_weat_effect_size(w, A, B):

    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=1, keepdims=True)

    A_associations = A_normed @ w_normed.T
    B_associations = B_normed @ w_normed.T
    joint_associations = np.concatenate((A_associations, B_associations), axis=1)

    effect_size = (np.mean(A_associations) - np.mean(B_associations)) / np.std(joint_associations, ddof=1)

    return effect_size

#Need to test p-value code
def weat(A, B, X, Y, permutations):

    A_normed = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=1, keepdims=True)
    X_normed = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normed = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    X_A = np.mean((X_normed @ A_normed.T), axis=1)
    X_B = np.mean((X_normed @ B_normed.T), axis=1)
    X_Associations = X_A - X_B

    Y_A = np.mean((Y_normed @ A_normed.T),axis=1)
    Y_B = np.mean((Y_normed @ B_normed.T),axis=1)
    Y_Associations = Y_A - Y_B

    joint_associations = np.concatenate((X_Associations, Y_Associations), axis=1)

    X_Mean = np.mean(X_Associations)
    Y_Mean = np.mean(Y_Associations)
    test_statistic = X_Mean - Y_Mean
    effect_size = test_statistic / np.std(joint_associations,ddof=1) 

    midpoint = X_Associations.shape[0]
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in permutations])
    sample_associations = np.mean(sample_distribution[:, :midpoint]) - np.mean(sample_distribution[:, midpoint:])
    p_value = 1 - norm.cdf(test_statistic, np.mean(sample_associations), np.std(sample_associations, ddof=1))

    return effect_size, p_value

def weat_effect_size(A, B, X, Y):

    A_normed = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=1, keepdims=True)
    X_normed = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normed = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    X_A = np.mean((X_normed @ A_normed.T), axis=1)
    X_B = np.mean((X_normed @ B_normed.T), axis=1)
    X_Associations = X_A - X_B

    Y_A = np.mean((Y_normed @ A_normed.T),axis=1)
    Y_B = np.mean((Y_normed @ B_normed.T),axis=1)
    Y_Associations = Y_A - Y_B

    joint_associations = np.concatenate((X_Associations, Y_Associations), axis=1)

    X_Mean = np.mean(X_Associations)
    Y_Mean = np.mean(Y_Associations)
    test_statistic = X_Mean - Y_Mean
    effect_size = test_statistic / np.std(joint_associations,ddof=1) 

    return effect_size

def valnorm(A, B, target_embeddings, human_ratings):

    embedding_associations = [sc_weat_effect_size(w, A, B) for w in target_embeddings]

    return pearsonr(embedding_associations, human_ratings)