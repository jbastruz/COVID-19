data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] y;
  vector[N] day;
  vector[M] day_projected;
  int<lower=0> K; //population effects
  matrix[N,K] design_matrix_1;
  matrix[N,K-1] design_matrix_b;
  matrix[M,K] design_matrix_proj;
  matrix[M,K-1] design_matrix_b_proj;
}
parameters {
  
  vector[K] a;

  vector<lower=0>[K-1] b;
  real<lower=0> b_intercept;
  real<lower=0> sigma_b;
  
  // vector<lower=0>[K] c_intercept;
  vector[K] c_b;
  vector<lower=0>[K] c_a;
  vector<lower=0>[K] c_h;
  
  //real<lower=0> sigma_c;

  // vector<lower=0>[K_a] a_2;
  // real<lower=0> a_intercept;
  // real<lower=0> sigma_a;
  
  // vector[K_c] c_2;
  // 
  // real<lower=0> sigma_lockdown;
  // real<lower=0> sigma_tests;
  // real<lower=0> sigma_density;
  // real<lower=0> sigma_group;
  // real<lower=0> sigma_Urbanpop;
  
  real<lower=0> sigma;
}
transformed parameters{
  
  vector[N] a_index=  design_matrix_1 * a;
  vector[N] b_index = b_intercept + design_matrix_b * b * sigma_b;
  vector<lower=0>[N] c_index = design_matrix_1 * c_h + 0.1 ./ (1 + exp(design_matrix_1 * c_b .* ( day - design_matrix_1 * c_a)));

  vector[N] mu= a_index .* exp(- b_index .* exp(-c_index .* day));
//  vector[N] sigma= design_matrix_sigma * sigma;

  vector[M] a_index_proj= design_matrix_proj * a;
  vector[M] b_index_proj= b_intercept + design_matrix_b_proj * b * sigma_b;
  vector<lower=0>[M] c_index_proj = design_matrix_proj * c_h + 0.1 ./ (1 + exp(design_matrix_proj * c_b .* (day_projected - design_matrix_proj * c_a)));

  
  vector[M] mu_proj= a_index_proj .* exp(-b_index_proj .* exp(-c_index_proj .* day_projected));
}
model {
//priors

target+= normal_lpdf(a|11,5);

target+= normal_lpdf(b|0,1);
target+= normal_lpdf(b_intercept|5,10);
target+= cauchy_lpdf(sigma_b|0,25);

// target+= normal_lpdf(c|c_intercept,sigma_c); //+ design_matrix_level_2 * c_2
// target+= normal_lpdf(c_intercept|0.1,0.5);
// target+= cauchy_lpdf(sigma_c|0,1);

target+= normal_lpdf(c_b|0,0.1);
target+= normal_lpdf(c_a[1]|0,10);
target+= normal_lpdf(c_a[2]|80,10);
target+= normal_lpdf(c_a[3]|80,10);
target+= normal_lpdf(c_h[1]|0.13,0.05);
target+= normal_lpdf(c_h[2]|0.03,0.02);
target+= normal_lpdf(c_h[3]|0.03,0.02);

// target+= normal_lpdf(c_2[1]*sigma_lockdown|0,1);
// target+= normal_lpdf(c_2[2]*sigma_tests|0,1);
// target+= normal_lpdf(c_2[3]*sigma_density|0,1);
// target+= normal_lpdf(c_2[4]*sigma_group|0,1);
// target+= normal_lpdf(c_2[5]*sigma_Urbanpop|0,1);
// 
// target+= cauchy_lpdf(sigma_lockdown|0,1);
// target+= cauchy_lpdf(sigma_tests|0,1);
// target+= cauchy_lpdf(sigma_density|0,1);
// target+= cauchy_lpdf(sigma_group|0,1);
// target+= cauchy_lpdf(sigma_Urbanpop|0,1);


target+= normal_lpdf(sigma|0,10);

target+= lognormal_lpdf(y|mu,sigma);
}
generated quantities{
 real yrep[N]= lognormal_rng(mu,sigma);
 real yproj[M]= lognormal_rng(mu_proj,sigma);
}
