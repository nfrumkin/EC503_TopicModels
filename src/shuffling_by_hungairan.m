function [beta_syn_reordered,cost]=shuffling_by_hungairan(beta,beta_hat)

Ntopic=size(beta,2);
for iTopic=1:Ntopic
    cost_M(:,iTopic)=sum(abs(beta_hat-beta(:,iTopic)))';  %l1 dist
end

[assignment,cost] = munkres(cost_M);

for iTopic=1:Ntopic
    beta_syn_reordered(:,iTopic)=beta_hat(:,assignment(iTopic));
    
end
    