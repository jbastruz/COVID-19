data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] y;
  vector[N] day;
  vector[M] day_projected;
  int<lower=0> J; //number of countries
  int<lower=0> L; //number of clusters
  int<lower=0> C; //number of c levels
  matrix[N,J] design_matrix_1;
  matrix[J,L] design_matrix_2;
  matrix[N,J-1] design_matrix_b;
  matrix[N,C] design_matrix_c;
  matrix[M,J] design_matrix_proj;
  matrix[M,J-1] design_matrix_b_proj;
}
parameters {
  
  vector<lower=0>[J] a_j;

  vector<lower=0>[J-1] b_j;
  real<lower=0> b_intercept;
  real<lower=0> sigma_b;
  
  vector<lower=0>[C] c_day;
  vector<lower=0>[J] c_country;
  vector<lower=0>[L] c_cluster;

  real<lower=0> sigma_c;  
  real<lower=0> sigma_c_country;
  real<lower=0> sigma_cluster;
  
  real<lower=0> sigma;
}
transformed parameters{
  
  vector[N] a_index=  design_matrix_1 * a_j;
  vector[N] b_index = b_intercept + design_matrix_b * b_j * sigma_b;
  vector<lower=0>[N] c_index = design_matrix_c * c_day;

  vector[N] mu= a_index .* exp(- b_index .* exp(-c_index .* day));

  vector[M] a_index_proj= design_matrix_proj * a_j;
  vector[M] b_index_proj= b_intercept + design_matrix_b_proj * b_j * sigma_b;
  vector<lower=0>[M] c_index_proj = design_matrix_proj * c_country;
  
  vector[M] mu_proj= a_index_proj .* exp(-b_index_proj .* exp(-c_index_proj .* day_projected));
}
model {
//priors

target+= normal_lpdf(a_j|11,6);

target+= normal_lpdf(b_j|0,1);
target+= normal_lpdf(b_intercept|5,10);
target+= cauchy_lpdf(sigma_b|0,25);

target+= normal_lpdf(c_day|design_matrix_1 * c_country,sigma_c);

target+= cauchy_lpdf(sigma_c|0,1);
target+= normal_lpdf(c_country|design_matrix_2 * c_cluster,sigma_c_country);

target+= normal_lpdf(c_cluster|0,sigma_cluster);
target+= cauchy_lpdf(sigma_c_country|0,0.5);
target+= cauchy_lpdf(sigma_cluster|0,0.5);

target+= normal_lpdf(sigma|0,10);

target+= lognormal_lpdf(y|mu,sigma);
}
generated quantities{
 real yrep[N]= lognormal_rng(mu,sigma);
 real yproj[M]= lognormal_rng(mu_proj,sigma);
}