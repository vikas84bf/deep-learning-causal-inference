library(keras)
install_keras()

# Follow simulated data of Wager and Athey 2018 -
# Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.
# 4 input variables, of which only the third affects propensity score.

N <- 1000
X <- matrix(runif(N*4),ncol=4)
p <- function(x){return( (1 + dbeta(x[3],2,4)) / 4 )}

D <- rbinom(N,1,apply(X,1,p))

trsize <- 3*N/4

training_x <- X[1:trsize,]
training_D <- D[1:trsize]
test_x <-  X[-(1:trsize),]
test_D <- D[-(1:trsize)]

# DNN of fixed width = 10+2d
# following Farrel et al 2018 (Deep Neural Networks for Estimation and Inference) Corollary 2

model <- keras_model_sequential() %>%
  layer_dense(units = 18, activation = 'relu', input_shape = c(4)) %>% #first hidden layer
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 18, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 18, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 18, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid') #output layer. Expit (sigmoid) output fn converts log(p/1-p) to p

compile(model, optimizer='sgd', loss='binary_crossentropy',metrics='acc') #logistic loss = binary_crossentropy
fit(model, training_x,training_D)

summary(predict(model,test_x)) # True propensity has Min = 0.2500, Max = 0.7773.
# ISSUE: dropout flattens the peak propensity's distribution severely (see next plot)

p_2 <- function(u){return( (1 + dbeta(u,2,4)) / 4 )} #rewrite p(x) to accept 1-dim input for plot
u <- seq(0,1,by=0.0001)
plot(u,p_2(u)) #true propensity score
points(test_x[,3],predict(model,test_x),col=2) #overlay model's predictions
