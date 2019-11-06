# Sequential Learning in Univariate DCMM

0. Set $m_0,C_0,G_t$ (see example 4.2), $W_t$ (see supplementary material).

## Bernoulli logistic DGLM

1. At any time $t-1$, current information is summarized as $(\theta_{t-1}|D_{t-1},I_{t-1})\sim(m_{t-1},C_{t-1})$

2. Evolutionary equation induces 1-step ahead prior: $(\theta_{t}|D_{t-1},I_{t-1})\sim(a_{t},R_{t}), a_t=G_tm_{t-1},R_t=G_tC_{t-1}G_t^T+W_t$

3. Conjugate prior: $\pi_t\sim Beta(\alpha_t,\beta_t)$ (no action)

4. Solve $(\alpha_t,\beta_t)$  from  $F_t 'a_t=:f_{t}=\psi\left(\alpha_{t}\right)-\psi\left(\beta_{t}\right), F_t' R_tF_t=:q_{t}=\psi^{\prime}\left(\alpha_{t}\right)+\psi^{\prime}\left(\beta_{t}\right)$, 

   where $\psi​$ and $\psi'​$ are digamma and trigamma function

5. 1-step ahead forecast: $\operatorname{Pr}\left[z_{t}=1 | \mathcal{D}_{t-1}, \mathcal{I}_{t-1}\right]=\alpha_{t} /\left(\alpha_{t}+\beta_{t}\right)​$. Here we make a prediction with $I(Ez_t>0.5)​$.

6. On observing $z_t​$, we have the posterior. (no action)

7. Calculate $g_{t}=\psi\left(\alpha_{t}+z_{t}\right)-\psi\left(\beta_{t}+1-z_{t}\right), p_{t}=\psi^{\prime}\left(\alpha_{t}+z_{t}\right)+\psi^{\prime}\left(\beta_{t}+1-z_{t}\right)​$ ($z_t​$ is observed!)

8. Posterior update $\left(\boldsymbol{\theta}_{t} | \mathcal{D}_{t}\right) \sim\left(\mathbf{m}_{t}, \mathbf{C}_{t}\right)$:

   $$\begin{array}{l}{\mathbf{m}_{t}=\mathbf{a}_{t}+\mathbf{R}_{t} \mathbf{F}_{t}\left(g_{t}-f_{t}\right) / q_{t} \quad \text { and }} \\ {\mathbf{C}_{t}=\mathbf{R}_{t}-\mathbf{R}_{t} \mathbf{F}_{t} \mathbf{F}_{t}^{\prime} \mathbf{R}_{t}^{\prime}\left(1-p_{t} / q_{t}\right) / q_{t}}\end{array}$$

(similar to Kalman filtering)

## Poisson log DGLM

1. At any time $t-1$, current information is summarized as $(\theta_{t-1}|D_{t-1},I_{t-1})\sim(m_{t-1},C_{t-1})$

2. Evolutionary equation induces 1-step ahead prior: $(\theta_{t}|D_{t-1},I_{t-1})\sim(a_{t},R_{t}), a_t=G_tm_{t-1},R_t=G_tC_{t-1}G_t^T+W_t$

3. Conjugate prior: $\mu_t\sim Gamma(\alpha_t,\beta_t)​$ (no action)

4. Solve $(\alpha_t,\beta_t)​$  from  $F_t 'a_t=:f_{t}=\psi\left(\alpha_{t}\right)-\log\left(\beta_{t}\right), F_t' R_tF_t=:q_{t}=\psi^{\prime}\left(\alpha_{t}\right)​$, 

   where $\psi$ and $\psi'$ are digamma and trigamma function

5. 1-step ahead forecast: $(y_t|D_{t-1},I_{t-1})\sim NB(\alpha_t,\frac{1}{1+\beta_t})$. (The notation here is same as Wikipedia for Nb, which is different from appendix of BerryWest2019)

   We can simply use the fact that $(y_t|z_t=1)\sim Poi(\mu_t)+1$, so our 1-step ahead forecast is $E(y_t|z_t=1,D_{t-1},I_{t-1})=E(\mu_t)+1=\alpha_t/\beta_t+1$ by iterated expectation.

6. On observing $z_t$, we have the posterior. (no action)

7. Calculate $g_{t}=\psi\left(\alpha_{t}+z_{t}\right)-\psi\left(\beta_{t}+1-z_{t}\right), p_{t}=\psi^{\prime}\left(\alpha_{t}+z_{t}\right)+\psi^{\prime}\left(\beta_{t}+1-z_{t}\right)$ ($z_t$ is observed!)

8. Posterior update $\left(\boldsymbol{\theta}_{t} | \mathcal{D}_{t}\right) \sim\left(\mathbf{m}_{t}, \mathbf{C}_{t}\right)$:

   $$\begin{array}{l}{\mathbf{m}_{t}=\mathbf{a}_{t}+\mathbf{R}_{t} \mathbf{F}_{t}\left(g_{t}-f_{t}\right) / q_{t} \quad \text { and }} \\ {\mathbf{C}_{t}=\mathbf{R}_{t}-\mathbf{R}_{t} \mathbf{F}_{t} \mathbf{F}_{t}^{\prime} \mathbf{R}_{t}^{\prime}\left(1-p_{t} / q_{t}\right) / q_{t}}\end{array}$$



## Filtering and Forecasting

In DGLM, The positive count model component will be updated ==only when $z_t=1​$.== When $z_t=0​$, the positive count value is treated as missing. I.e., update both Poisson and Bernoulli when $z_t=1​$, and update only Bernoulli when $z_t=0​$.

Forward filtering: Bernoulli and Poisson are updated separately.

Forecasting: 

**$t+k$: Implied mixture of Bernoulli and shifted Poisson (step 5 above, Appendix 3)**

At time *t*, the *k*-step ahead forecast distribution has a pdf of the compositional form

$$ p\left(y_{t+k} | \mathcal{D}_{t}, \mathcal{I}_{t}, \pi_{t+k}\right)=\left(1-\pi_{t+k}\right) \delta_{0}\left(y_{t+k}\right) +\pi_{t+k} h_{t, t+k}\left(y_{t+k}\right) $$

where 

![image-20191105182726257](/Users/zhuoqunwang/Library/Application Support/typora-user-images/image-20191105182726257.png)

**Full joint forecast of $y_{t+1:t+k}$: Monte Carlo samples from $p(y_{t+1:t+k}|D_{t},I_t)$ (Appendix 3)**







Try: (Maybe a "future possibilities" part in the presentation on Nov 19)

1. Normal DLM for simulated continuous data
2. performance of this model when it is not the true model (via simulation). e.g., when the true data generating process is simply poisson
3. performance of this model when the counts are not sparse and are large (so maybe it can be treated as continuous): e.g., 中国进出口总额数据, and compare it to normal DGLM or ARMA
4. $W_t$: no need to specify it carefully according to supplementary material. Try different specifications and look at their influence on the outcome (maybe in the univariate case) 
5. different loss functions 



