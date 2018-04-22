% Hung-Chen Yu

%{
doc: a text corpus in bag of word matrix  [ D x W ]
beta: probability of words for each topic [ W x T ]
%}


function [T_C_avg,T_C]=UMass_topic_coherence(doc,beta,Num_Top_w,Epsilon)

if ~exist('Epsilon','var')
    Epsilon=1;
end

Ndocs=size(doc,1);
Ntopics=size(beta,2);
Nwords=size(beta,1);


[~,top_word_ID]=sort(beta,'descend');

doc_bool=double(doc~=0);
doc_freq=sum(doc_bool);


T_C=zeros(Ntopics,1);
reverseStr=[];
for itopic=1:Ntopics
    msg=sprintf('Calculating Topic..... %d  / %d  \n', itopic,Ntopics);
    fprintf([reverseStr,msg]);
    
    top_word_ID_list=top_word_ID(1:Num_Top_w,itopic);
    
    
    for iw=2:Num_Top_w
        for iw_paired= 1 : (iw-1)
            doc_co_freq=doc_bool(:,top_word_ID_list(iw))'*doc_bool(:,top_word_ID_list(iw_paired));
            T_C(itopic)=T_C(itopic)+log10( (doc_co_freq+Epsilon)/doc_freq(top_word_ID_list(iw)));
        end
    end
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end

T_C_avg=mean(T_C);

disp('--------------- Finish calculating the topic coherence ---------------');
end

                
        


