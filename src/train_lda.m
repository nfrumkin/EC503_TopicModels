clear
clc all;

% load data
fprintf("======= load data\n")
[vocab] = textread('../../data/vocab.txt','%s');
[doc_id_train, word_id_train, wc_train] = textread('../../data/docword.nytimes.txt','%d %d %d');


% trim away top lines
doc_id_train = doc_id_train(4:end, :);
word_id_train = word_id_train(4:end, :);
wc_train = wc_train(4:end, :);

num_vocab = length(vocab);
num_docs = length(unique(doc_id_train));

% convert data to count matrix, need to use batches so matlab can handle
% data
fprintf("======= create counts matrix")
batch_size =  int32(num_docs/100);
counts_matrix = sparse(doc_id_train, word_id_train, wc_train);


num_topics = [10;20;50;100;200];
for i = 1:4
    i
    fprintf("======= generate LDA model")
    model_cgs(i) = fitlda(counts_matrix(1:50000,:), num_topics(i,1), "Solver","cgs");
end

num_topics = [10;20;50;100;200];
for i = 1:4
    i
    fprintf("======= generate LDA model")
    model_cgs(i) = fitlda(counts_matrix(1:50000,:), num_topics(i,1), "Solver","avb");
end

generate word cloud for visualization
figure
test_doc = 3957;

for i = 1:5
    [topic_pred, num] = predict(model(1),counts_matrix(test_doc,:))
    topic_words = vocab(str2double(model_avb.Vocabulary));
    for topicIdx = topic_pred(1)%1:num_topics
        subplot(3,2,i)
        wordcloud(topic_words,model_avb(i).TopicWordProbabilities(:,topicIdx));
        title("Number of Topics: " + num_topics(i))
    end
end

subplot(3,2,6)
[B, I] = maxk(counts_matrix(test_doc,:),10);
bar(B)
title("Most Frequent Words in Document")
ylabel("Frequency")
set(gca,'xticklabel', vocab(I))
xtickangle(45);

