% Compare with different solver

Ndoc=10000;
Nterm_doc=135;  %avg of the original data
numTopics=40; %40

load('synthetic_doc_cgs_10000doc.mat');


max_iter_train= 100   ;
[pz, pz_d_train, beta_hat_plsa, perplex_train]=fit_plsa(synthetic_doc_cgs_10000doc, numTopics, max_iter_train)

[beta_syn_reordered_plsa,cost_plsa]=shuffling_by_hungairan(beta_ground_truth, beta_hat_plsa);

save('test_result_plsa.mat','pz', 'pz_d_train', 'beta_hat_plsa', 'perplex_train','beta_syn_reordered_plsa','cost_plsa');
