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


[~,top_word_ID]=sort(beta);

for iw=1:Nwords
    doc_freq(iw)=length(find(doc(:,iw)));
end


T_C=zeros(Ntopics,1);

reverseStr=[];
for itopic=1:Ntopics
    msg=sprintf('Calculating Topic..... %d  / %d  \n', itopic,Ntopics);
    fprintf([reverseStr,msg]);
    top_word_ID_list=top_word_ID(1:Num_Top_w,itopic);
    
    
    doc_freq=zeros(Num_Top_w,1);
    doc_freq(1)=length( find(doc(:, top_word_ID_list(1)) ) );
    for iw=2:Num_Top_w
        doc_freq(iw)=length( find(doc(:, top_word_ID_list(iw)) ) );
        
        for iw_paired= 1 : (iw-1)
            doc_co_freq=0;
            for idoc=1:Ndocs
                if (doc(idoc, top_word_ID_list(iw)) *  doc(idoc, top_word_ID_list(iw_paired)))~=0
                    doc_co_freq=doc_co_freq+1;
                end
            end
            T_C(itopic)=T_C(itopic)+log10( (doc_co_freq+Epsilon)/doc_freq(iw));
        end
    end
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end

T_C_avg=mean(T_C);

end

                
        


