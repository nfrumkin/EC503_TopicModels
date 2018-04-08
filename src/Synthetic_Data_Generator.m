%% Synthetic Data Generator
%{
Output: Bag of Word Matrix with the same WordID as the one on inputed lda_model


Ntopic: Total number of topic in the dataset
Nvocabulary: Total number of unqiue vocabulary in dataset
Ndoc: Total number of documents in the dataset
Nterm_doc: Number of terms per document (can be a vector)
%}

function [synthetic_doc,theta]=Synthetic_Data_Generator(lda_model, Ndoc, Nterm_doc, alpha)
if ~exist('alpha','var')
    alpha=0.03*ones(1,lda_model.NumTopics);
end

if isscalar(Nterm_doc)
    Nterm_doc=ones(Ndoc,1)*Nterm_doc;
end

Nvocabulary=size(lda_model.Vocabulary,2);

synthetic_doc=zeros(Ndoc,Nvocabulary);
for idoc=1:Ndoc
   
    theta(idoc,:)=drchrnd(alpha,1); %sample form dirchlet distribution
    
    Nterm_topic_doc=mnrnd(Nterm_doc(idoc),theta(idoc,:));
    for iTopic=1:lda_model.NumTopics
        word_count=mnrnd(Nterm_topic_doc(iTopic) , lda_model.TopicWordProbabilities(:,iTopic));
        synthetic_doc(idoc,:)=synthetic_doc(idoc,:)+word_count;
    end     
    
    word_idx=find(any(synthetic_doc,1));
    
    
end



end


% https://cxwangyi.wordpress.com/2009/03/18/to-generate-random-numbers-from-a-dirichlet-distribution/
function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);

end
