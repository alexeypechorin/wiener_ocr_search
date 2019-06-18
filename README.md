Files for searching through wiener: 

- large_scale_search.py must be run first, to create the hub matrix for the scaling for nearest neighbors to work. 
- everything will be saved in model_data_deu 
- then we can use the files saved to run any query 


FILE EXPLANATIONS in model_data_deu: 

- candidates_all.npy: the L2 normalized candidates matrix after projection using CCA i.e. normalize((PHOC - mean_y)*Wy)
- Wy.npy: the projection matrix found using CCA to project candidate PHOC matrix to common subspace 
- mean_y.npy: mean vector used in projecting candidate PHOC matrix to common subspace
- Wx.npy: projection matrix found using CCA to project PHOC query to commmon subspace
- mean_x.npy: mean vector used in projecting PHOC query to common subspace. Using Wx, and mean_x, we will project each candidate like (candidatePHOC - mean_x)*Wx
- hub.npy: average distance to 20 nearest neighbors for the normalized candidates
- phoc_candidates.npy: the candidates before CCA projection and normalization
- unigrams.json: the character set used to create the PHOC vectors
- words.json: list of dictionaries where each dictionary is a unique word with three keys properties --> 'word', 'bbox', and 'img_path' 
- vocabulary.json: dictionary where each entry is a unique word, and the value are the indices where the given word appears in the words.json list
- vocab_strings.json: just the keys (the words) of the vocabulary.json dictionary as a list 
- thus the n-th row of candidates_all.npy contains the word at the n-th index of vocab_strings.json



SIMPLE SEARCH: 
- search_user_input.py is a simple function allowing search through user input from the command line and the results are stored in results_user_input
