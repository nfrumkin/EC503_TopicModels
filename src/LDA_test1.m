clc
clear all
close all

data=readtable('test1.csv');
text=data.Var10;
text=erasePunctuation(text);
doc=tokenizedDocument(text);
doc_bag = bagOfWords(doc);
doc_bag = removeWords(doc_bag,stopWords); %matlab stopword list
%training  % https://www.mathworks.com/help/textanalytics/ref/fitlda.html
numTopics=4;
lda_model = fitlda(doc_bag,numTopics);


Ndoc=10;
Nterm_doc=1000;
[doc_sythetic,theta]=Synthetic_Data_Generator(lda_model, Ndoc, Nterm_doc);
doc_sythetic=tokenizedDocument(doc_sythetic);
doc_sythetic_bag = bagOfWords(doc_sythetic);
lda_model_sythetic = fitlda(doc_bag,numTopics);


% figure
% for topicIdx = 1:numTopics
%     subplot(2,2,topicIdx)
%     wordcloud(mdl,topicIdx);
%     title("Topic: " + topicIdx)
% end

%predict
% newDocuments = tokenizedDocument([
%     "what's in a name? a rose by any other name would smell as sweet."
%     "if music be the food of love, play on."]);
% topicIdx = predict(mdl,newDocuments)
% 
% 
% 
% figure
% subplot(1,2,1)
% wordcloud(mdl,topicIdx(1));
% title("Topic " + topicIdx(1))
% subplot(1,2,2)
% wordcloud(mdl,topicIdx(2));
% title("Topic " + topicIdx(2))

