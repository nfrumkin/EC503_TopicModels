clear
clc all;

% load data
[vocab] = textread('../data/vocab.txt','%s');
[doc_id_train, word_id_train, wc_train] = textread('../data/docword.nytimes.txt','%d %d %d');

% trim away top lines
doc_id_train = doc_id_train(4:end, :);
word_id_train = word_id_train(4:end, :);
wc_train = wc_train(4:end, :);

num_vocab = length(vocab);
num_docs = length(unique(doc_id_train));

% allocate sparse matrix for counts
counts_matrix = spalloc(num_docs, num_vocab,num_docs*1000);

% convert data to count matrix, need to use batches so matlab can handle
% data
batch_size =  int32(num_docs/100);
for i = 1:100
   i
   
   % convert doc id and word id to row and col vals
   ind = sub2ind(size(counts_matrix), doc_id_train((i-1)*batch_size+1:i*batch_size), word_id_train((i-1)*batch_size+1:i*batch_size));
   
   % index into counts matrix and place appropriate word count
   counts_matrix(ind) = wc_train((i-1)*batch_size+1:i*batch_size);
end

num_topics = 10;
mdl = fitlda(counts_matrix(1:15000,:), num_topics);

% generate word cloud for visualization
figure
topic_words = vocab(str2double(mdl.Vocabulary));
for topicIdx = 1:num_topics
    subplot(2,2,topicIdx)
    wordcloud(topic_words,mdl.TopicWordProbabilities(:,topicIdx));
    title("Topic: " + topicIdx)
end