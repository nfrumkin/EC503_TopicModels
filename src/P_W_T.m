load("../../models/model_vb_10.mat");
beta = model_avb_10.DocumentTopicProbabilities;
hm_array(1) = harmonic_mean(beta);

load("../../models/model_vb_20.mat");
beta = model_avb_20.DocumentTopicProbabilities;
hm_array(2) = harmonic_mean(beta);

load("../../models/model_vb_50.mat")
beta = model_avb_50.DocumentTopicProbabilities;
hm_array(3) = harmonic_mean(beta);


load("../../models/model_vb_100.mat");
beta = model_avb_100.DocumentTopicProbabilities;
hm_array(4) = harmonic_mean(beta);


topics = [10,20,50,100];
bar(hm_array)
set(gca, 'xticklabel', topics)
xlabel("number of topics")
ylabel("P(w | T)")
title("P(w | T) for Variational Bayes Method")

function hm = harmonic_mean(input_beta)
    [num_data, num_topics] = size(input_beta);
    inv_beta = 1./input_beta;
    hm = num_topics/(sum(sum(inv_beta,2)));
end

