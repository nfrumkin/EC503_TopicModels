%% Synthetic Data Generator
%{
Ntopic: Total number of topic in the dataset
Nvocabulary: Total number of unqiue vocabulary in dataset
Ndoc: Total number of documents in the dataset
Nterm_doc: Number of terms per document (can be a vector)
%}

function [doc,theta]=Synthetic_Data_Generator(lda_model, Ndoc, Nterm_doc, alpha)
if ~exist('alpha','var')
    alpha=0.03*ones(1,lda_model.NumTopics);
end

if isscalar(Nterm_doc)
    Nterm_doc=ones(Ndoc,1)*Nterm_doc;
end

Nvocabulary=size(lda_model.Vocabulary,2);

doc=repmat({[' ']},Ndoc,1);

for idoc=1:Ndoc
   
    theta(idoc,:)=drchrnd(alpha,1); %sample form dirchlet distribution
    synthetic_doc=zeros(1,Nvocabulary);
    for iTopic=1:lda_model.NumTopics
        Nterm_topic_doc=round(Nterm_doc(idoc)*theta(iTopic));
        word_count=mnrnd(Nterm_topic_doc , lda_model.TopicWordProbabilities(:,iTopic));
        synthetic_doc=synthetic_doc+word_count;
    end     
    
    word_idx=find(any(synthetic_doc,1));
    
    
    for i=1:length(word_idx)
        repeat_times=synthetic_doc(word_idx);
        for j=1:repeat_times
            doc(idoc)=strcat( doc(idoc), {char(lda_model.Vocabulary(word_idx(i)))},{' '}  );
        end
    end
    
end

doc=string(doc);

end

function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);

end
