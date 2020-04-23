data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] y;
  vector[N] day;
  vector[M] day_projected;
  int<lower=0> J; //number of countries
  int<lower=0> L; //number of clusters
  matrix[N,J] design_matrix_1;
  matrix[J,L] design_matrix_2;
  matrix[N,J-1] design_matrix_b;
  matrix[M,J] design_matrix_proj;
  matrix[M,J-1] design_matrix_b_proj;
}
parameters {
  
  vector[J] a_j;

  vector<lower=0>[J-1] b_j;
  real<lower=0> b_intercept;
  real<lower=0> sigma_b;
  
  vector<lower=0>[J] c_j;
  vector<lower=0>[J] c_h;
  vector<lower=0>[J] c_b;
  vector<lower=0>[J] c_a;
  real<lower=0> sigma_c_j;  
  real<lower=0> sigma_c_h;  
  real<lower=0> sigma_c_b;  
  real<lower=0> sigma_c_a; 
  
  vector<lower=0>[L] c_cluster_1;
  vector<lower=0>[L] c_cluster_2;
  vector<lower=0>[L] c_cluster_3;  
  vector<lower=0>[L] c_cluster_4;  

  real<lower=0> sigma;
}
transformed parameters{
  
  vector[N] a_index=  design_matrix_1 * a_j;
  vector[N] b_index = b_intercept + design_matrix_b * b_j * sigma_b;
  vector<lower=0>[N] c_index = design_matrix_1 * c_j ./ (design_matrix_1 * c_h + exp(-design_matrix_1 * c_b .* (day - design_matrix_1 * c_a)));

  vector[N] mu= a_index .* exp(- b_index .* exp(-c_index .* day));
//  vector[N] sigma= design_matrix_sigma * sigma;

  vector[M] a_index_proj= design_matrix_proj * a_j;
  vector[M] b_index_proj= b_intercept + design_matrix_b_proj * b_j * sigma_b;
  vector<lower=0>[M] c_index_proj = design_matrix_proj * c_j ./ (design_matrix_proj * c_h + exp(-design_matrix_proj * c_b .* (day_projected- design_matrix_proj *c_a )));

  
  vector[M] mu_proj= a_index_proj .* exp(-b_index_proj .* exp(-c_index_proj .* day_projected));
}
model {
//priors

target+= normal_lpdf(a_j|11,6);

target+= normal_lpdf(b_j|0,1);
target+= normal_lpdf(b_intercept|5,10);
target+= cauchy_lpdf(sigma_b|0,25);

target+= normal_lpdf(c_j|design_matrix_2 * c_cluster_1,sigma_c_j);
target+= normal_lpdf(c_h|design_matrix_2 * c_cluster_2,sigma_c_h);
target+= normal_lpdf(c_b|design_matrix_2 * c_cluster_3,sigma_c_b);
target+= normal_lpdf(c_a|design_matrix_2 * c_cluster_4,sigma_c_a);

target+= normal_lpdf(c_cluster_1|0,1);
target+= normal_lpdf(c_cluster_2|0,10);
target+= normal_lpdf(c_cluster_3|0,0.5);
target+= normal_lpdf(c_cluster_4|80,20);

target+= cauchy_lpdf(sigma_c_j|0,1);
target+= cauchy_lpdf(sigma_c_h|0,10);
target+= cauchy_lpdf(sigma_c_b|0,1);
target+= cauchy_lpdf(sigma_c_a|0,70);

target+= normal_lpdf(sigma|0,10);

target+= lognormal_lpdf(y|mu,sigma);
}
generated quantities{
 real yrep[N]= lognormal_rng(mu,sigma);
 real yproj[M]= lognormal_rng(mu_proj,sigma);
}
