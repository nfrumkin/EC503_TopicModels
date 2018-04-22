% Compare with different solver

%% Generating syn data
Ndoc=10000;
Nterm_doc=135;  %avg of the original data
numTopics=40; %40

load('model_cgs.mat');

[synthetic_doc_cgs,theta_cgs]=Synthetic_Data_Generator(model_cgs(numTopics/10), Ndoc, Nterm_doc);
synthetic_doc_cgs_10000doc=sparse(synthetic_doc_cgs);
save('synthetic_doc_cgs_10000doc.mat','synthetic_doc_cgs_10000doc','theta_cgs');

synthetic_doc=synthetic_doc_cgs_10000doc;
clear synthetic_doc_cgs synthetic_doc_cgs_10000doc;

beta_ground_truth=model_cgs(numTopics/10).TopicWordProbabilities;

%%
lda_model_sythetic_cgs = fitlda(synthetic_doc,numTopics,'Solver','cgs');

[beta_syn_reordered_cgs,cost_cgs]=shuffling_by_hungairan(beta_ground_truth, lda_model_sythetic_cgs.TopicWordProbabilities);

save('test_result_cgs.mat','lda_model_sythetic_cgs','beta_syn_reordered_cgs','cost_cgs');

%%

lda_model_sythetic_savb = fitlda(synthetic_doc,numTopics,'Solver','savb');

[beta_syn_reordered_savb,cost_savb]=shuffling_by_hungairan(beta_ground_truth, lda_model_sythetic_savb.TopicWordProbabilities);

save('test_result_savb.mat','lda_model_sythetic_savb','beta_syn_reordered_savb','cost_savb');

%%
lda_model_sythetic_avb = fitlda(synthetic_doc,numTopics,'Solver','avb');

[beta_syn_reordered_avb,cost_avb]=shuffling_by_hungairan(beta_ground_truth, lda_model_sythetic_avb.TopicWordProbabilities);

save('test_result_avb.mat','lda_model_sythetic_avb','beta_syn_reordered_avb','cost_avb');

%%
lda_model_sythetic_cvb0 = fitlda(synthetic_doc,numTopics,'Solver','cvb0');

[beta_syn_reordered_cvb0,cost_cvb0]=shuffling_by_hungairan(beta_ground_truth, lda_model_sythetic_cvb0.TopicWordProbabilities);

save('test_result_cvb0.mat','lda_model_sythetic_cvb0','beta_syn_reordered_cvb0','cost_cvb0');