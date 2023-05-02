import numpy as np
from numpy.linalg import inv, qr, eig, norm
import math
from math import isclose, sqrt
#from tqdm import tqdm
import time
import tensorflow as tf
tf.reset_default_graph()

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--storage', '-m', default=10, help='The Memory Storage')
parser.add_argument('--mini_batch','-minibatch', default=1000,help='minibatch size')
parser.add_argument('--num_batch_in_data', '-num-batch',default=5,
        							help='number of batches with overlap')
parser.add_argument('--method', '-method',default='trust-region',
        	help="""Method of optimization ['line-search','trust-region']""")
parser.add_argument(
        '--whole_gradient','-use-whole-data', action='store_true',default=False,
        help='Compute the gradient using all data')
parser.add_argument('--max_iter', '-maxiter', default=200, help='max iterations')

args = parser.parse_args()

minibatch = int(args.mini_batch)
m = int(args.storage)
num_batch_in_data = int(args.num_batch_in_data)
use_whole_data = args.whole_gradient
# if minibatch==500: ==> num_batch_in_data in [3, 6, 9, 12, 18, 36, 54, 108]
# if minibatch==1000 ==> num_batch_in_data in [3, 6, 9, 18, 54]
# if minibatch ==540 ==> num_batch_in_data in [5, 10, 20, 25, 50, 100]
# if minibatch ==1080 ==> num_batch_in_data in [5, 10, 25, 50]
method = str(args.method)
# ['line-search','trust-region']
max_num_iter = int(args.max_iter)

iter_num = 0
###############################################################################
######################## MNIST DATA ###########################################
###############################################################################
import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)
X_train, y_train = shuffle_data(data)
# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

X_test = data.test.images
y_test = data.test.labels

X_validation = data.validation.images
y_validation = data.validation.labels

X_train_multi = []
y_train_multi = []
###############################################################################
######################## LBFGS PARAMS #########################################
###############################################################################

S = np.array([[]])
Y = np.array([[]])
gamma = 1
lambda_min = 0
alpha = 1
# GLOBAL VARIABLES - MATRICES
P_ll = np.array([[]]) # P_parallel 
g_ll = np.array([[]]) # g_Parallel
g_NL_norm = 0
Lambda_1 = np.array([[]])

g = np.array([])
###############################################################################
######################## LeNet-5 Network Architecture #########################
###############################################################################
# number of weights and bias in each layer
n_W = {}
dim_w = {}

# network architecture hyper parameters
input_shape = [-1,28,28,1]
W0 = 28
H0 = 28

# Layer 1 -- conv
D1 = 1; F1 = 5; K1 = 20; S1 = 1
W1 = (W0 - F1) // S1 + 1
H1 = (H0 - F1) // S1 + 1
conv1_dim = [F1, F1, D1, K1]
conv1_strides = [1,S1,S1,1] 
n_W['1_w_conv'] = F1 * F1 * D1 * K1
n_W['1_b_conv'] = K1 
dim_w['1_w_conv'] = [F1, F1, D1, K1]
dim_w['1_b_conv'] = [K1]

# Layer 2 -- max pool
D2 = K1; F2 = 2; K2 = D2; S2 = 2
W2 = (W1 - F2) // S2 + 1
H2 = (H1 - F2) // S2 + 1
layer2_ksize = [1,F2,F2,1]
layer2_strides = [1,S2,S2,1]

# Layer 3 -- conv
D3 = K2; F3 = 5; K3 = 50; S3 = 1
W3 = (W2 - F3) // S3 + 1
H3 = (H2 - F3) // S3 + 1
conv2_dim = [F3, F3, D3, K3]
conv2_strides = [1,S3,S3,1] 
n_W['2_w_conv'] = F3 * F3 * D3 * K3
n_W['2_b_conv'] = K3 
dim_w['2_w_conv'] = [F3, F3, D3, K3]
dim_w['2_b_conv'] = [K3]

# Layer 4 -- max pool
D4 = K3; F4 = 2; K4 = D4; S4 = 2
W4 = (W3 - F4) // S4 + 1
H4 = (H3 - F4) // S4 + 1
layer4_ksize = [1,F4,F4,1]
layer4_strides = [1,S4,S4,1]


# Layer 5 -- fully connected
n_in_fc = W4 * H4 * D4
n_hidden = 500
fc_dim = [n_in_fc,n_hidden]
n_W['3_w_fc'] = n_in_fc * n_hidden
n_W['3_b_fc'] = n_hidden
dim_w['3_w_fc'] = [n_in_fc,n_hidden]
dim_w['3_b_fc'] = [n_hidden]
# Layer 6 -- output
n_in_out = n_hidden
n_W['4_w_fc'] = n_hidden * n_classes
n_W['4_b_fc'] = n_classes
dim_w['4_w_fc'] = [n_hidden,n_classes]
dim_w['4_b_fc'] = [n_classes]


for key, value in n_W.items():
	n_W[key] = int(value)

###############################################################################
######################## f(x;w) ###############################################
###############################################################################
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

w_initializer = tf.contrib.layers.xavier_initializer()

w_tf = {}
for key, _ in dim_w.items():
	w_tf[key] = tf.get_variable(key, shape=dim_w[key], 
											initializer=w_initializer)

def lenet5_model(x,_w):
	# Reshape input to a 4D tensor 
	x = tf.reshape(x, shape = input_shape)
	# LAYER 1 -- Convolution Layer
	conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
									filter =_w['1_w_conv'],
									strides = [1,S1,S1,1],
									padding = 'VALID') + _w['1_b_conv'])
	# Layer 2 -- max pool
	conv1 = tf.nn.max_pool(	value = conv1, 
							ksize = [1, F2, F2, 1], 
							strides = [1, S2, S2, 1], 
							padding = 'VALID')

	# LAYER 3 -- Convolution Layer
	conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
									filter =_w['2_w_conv'],
									strides = [1,S3,S3,1],
									padding = 'VALID') + _w['2_b_conv'])
	# Layer 4 -- max pool
	conv2 = tf.nn.max_pool(	value = conv2 , 
							ksize = [1, F4, F4, 1], 
							strides = [1, S4, S4, 1], 
							padding = 'VALID')
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer
	fc = tf.contrib.layers.flatten(conv2)
	fc = tf.nn.relu(tf.matmul(fc, _w['3_w_fc']) + _w['3_b_fc'])
	# fc = tf.nn.dropout(fc, dropout_rate)

	y_ = tf.matmul(fc, _w['4_w_fc']) + _w['4_b_fc']
	return y_

# Construct model
model = lenet5_model
y_ = model(x,w_tf)

# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################
######################## TF GRADINETS #########################################
###############################################################################
grad_w_tf = {}
for layer, _ in w_tf.items():
	grad_w_tf[layer] = tf.gradients(xs=w_tf[layer], ys=loss)

###############################################################################
######################## TF Auxilary variables ################################
###############################################################################
aux_w = {}
for layer, _ in w_tf.items():
	name = layer + 'aux_w_'
	aux_w[layer] = tf.get_variable(name=name, 
					shape=w_tf[layer].get_shape(), initializer=w_initializer)

aux_w_placeholder = {}
for layer, _ in w_tf.items():
	aux_w_placeholder[layer] = tf.placeholder(dtype="float",
										shape=w_tf[layer].get_shape())
aux_w_init = {}
for layer, _ in w_tf.items():
	aux_w_init[layer] = aux_w[layer].assign(aux_w_placeholder[layer])

aux_output = model(x,aux_w)
aux_loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = aux_output))
aux_grad_w = {}
for layer, _ in w_tf.items():
	aux_grad_w[layer] = tf.gradients(xs=aux_w[layer], ys=aux_loss)

update_w = {}
update_w_placeholder = {}
for layer, _ in w_tf.items():
	update_w_placeholder[layer] = tf.placeholder(dtype="float",
										shape=w_tf[layer].get_shape())
for layer, _ in w_tf.items():
	update_w[layer] = w_tf[layer].assign(update_w_placeholder[layer])

###############################################################################
###############################################################################
saver = tf.train.Saver()
init = tf.global_variables_initializer()
###############################################################################
###############################################################################

def backtracking_line_search(sess,g):
	alpha = 1
	rho_ls = 0.9
	c1 = 1E-4
	BLS_COND = False
	p = -g
	while not BLS_COND:
		new_f = eval_aux_loss(sess,alpha*p)
		old_f = eval_loss(sess)
		lhs = new_f
		rhs = old_f + c1 * alpha * p @ g
		BLS_COND = lhs <= rhs
		
		if BLS_COND:
			print('Backtracking line search satisfied for alpha = {0:.4f}' \
																.format(alpha))
		if alpha < 0.1:
			print('WARNING! Backtracking line search did not work')
			break
		alpha = alpha * rho_ls
	return alpha*p

def quad_model():
	pass

def phi_bar_func(sigma,delta):
	# phi(sigma) = 1 / v(sigma) - 1 / delta	
	u = sum( (g_ll ** 2) / ((Lambda_1 + sigma) ** 2) ) + \
									(g_NL_norm ** 2) / ( (gamma + sigma) ** 2)
	v = sqrt(u) 

	phi = 1 / v - 1 / delta
	return phi

def phi_bar_prime_func(sigma):
	u = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 2 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 2

	u_prime = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 3 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 3
	phi_bar_prime = u ** (-3/2) * u_prime

	return phi_bar_prime


def solve_newton_equation_to_find_sigma(delta):
	# tolerance
	tol = 1E-4
	sigma = max( 0, -lambda_min )
	if phi_bar_func(sigma,delta) < 0:
		sigma_hat = max( abs( g_ll ) / delta - Lambda_1 )
		sigma = max( 0, sigma_hat)
		while( abs( phi_bar_func(sigma,delta) ) > tol ):
			phi_bar = phi_bar_func(sigma,delta)
			phi_bar_prime = phi_bar_prime_func(sigma)
			sigma = sigma - phi_bar / phi_bar_prime
		sigma_star = sigma
	elif lambda_min < 0:
		sigma_star = - lambda_min
	else:
		sigma_star = 0

	return sigma_star 

def lbfgs_line_search_subproblem_solver(sess, g):
	# dimension of w
	n = sum(n_W.values())

	Psi = np.concatenate( (gamma*S, Y) ,axis=1)
	
	S_T_Y = S.T @ Y
	L = np.tril(S_T_Y,k=-1)
	U = np.tril(S_T_Y.T,k=-1).T
	D = np.diag( np.diag(S_T_Y) )

	M = - inv( np.block([ 	[gamma * S.T @ S ,	L],
							[     L.T,		   -D] 
			]) )

	Q, R = qr(Psi, mode='reduced')
	eigen_values, eigen_vectors = eig( R @ M @ R.T )

	# sorted eigen values
	idx = eigen_values.argsort()
	eigen_values_sorted = eigen_values[idx]
	eigen_vectors_sorted = eigen_vectors[:,idx]

	Lambda_hat = eigen_values_sorted
	V = eigen_vectors_sorted

	global P_ll
	global g_ll
	global g_NL_norm
	global Lambda_1

	Lambda_1 = gamma + Lambda_hat
	#Lambda_2 = gamma * np.ones( n-len(Lambda_hat) )
	#B_diag = np.concatenate( (Lambda_1, Lambda_2),axis=0 )
	global lambda_min
	lambda_min = min( Lambda_1.min(), gamma )
	
	P_ll = Psi @ inv(R) @ V # P_parallel 
	g_ll = P_ll.T @ g	# g_Parallel
	g_NL_norm = sqrt ( abs( norm(g) ** 2 - norm(g_ll) ** 2 ) )

	p = - 1 / gamma * \
			( g - Psi @ inv( gamma * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )

	alpha = satisfy_wolfe_condition(sess,p)

	return alpha * p


def satisfy_wolfe_condition(sess, p):
	alpha = 1
	rho_ls = 0.9
	c1 = 1E-4
	c2 = 0.9
	WOLFE_COND_1 = False
	WOLFE_COND_2 = False
	while not ( WOLFE_COND_1 and WOLFE_COND_2): 
		new_f = eval_aux_loss(sess,alpha*p)
		old_f = eval_loss(sess)
		lhs = new_f
		rhs = old_f + c1 * alpha * p @ g
		WOLFE_COND_1 = lhs <= rhs
		if WOLFE_COND_1:
			print('WOLFE_COND_1 SATISFIED')
		else:
			print('WOLFE_COND_1 NOT SATISFIED')

		new_g = eval_aux_gradient_vec(sess)
		lhs = new_g @ p
		rhs = c2 * g @ p
		WOLFE_COND_2 = lhs >= rhs
		if WOLFE_COND_2:
			print('WOLFE_COND_2 SATISFIED')
		else:
			print('WOLFE_COND_2 NOT SATISFIED')

		if WOLFE_COND_1 and WOLFE_COND_2:
			print('WOLFE CONDITIONS SATISFIED')
			print('alpha = {0:.4f}' .format(alpha))
		
		if alpha < 0.1:
			print('WARNING! Wolfe Condition did not satisfy')
			break
		alpha = alpha * rho_ls
	return alpha


def lbfgs_trust_region_subproblem_solver(delta, g):
	# size of w = g.size
	n = sum(n_W.values())

	Psi = np.concatenate( (gamma*S, Y) ,axis=1)
	
	S_T_Y = S.T @ Y
	L = np.tril(S_T_Y,k=-1)
	U = np.tril(S_T_Y.T,k=-1).T
	D = np.diag( np.diag(S_T_Y) )

	M = - inv( np.block([ 	[gamma * S.T @ S ,	L],
							[     L.T,		   -D] 
			]) )

	Q, R = qr(Psi, mode='reduced')
	eigen_values, eigen_vectors = eig( R @ M @ R.T )

	# sorted eigen values
	idx = eigen_values.argsort()
	eigen_values_sorted = eigen_values[idx]
	eigen_vectors_sorted = eigen_vectors[:,idx]

	Lambda_hat = eigen_values_sorted
	V = eigen_vectors_sorted

	global P_ll
	global g_ll
	global g_NL_norm
	global Lambda_1

	Lambda_1 = gamma + Lambda_hat
	#Lambda_2 = gamma * np.ones( n-len(Lambda_hat) )
	#B_diag = np.concatenate( (Lambda_1, Lambda_2),axis=0 )


	P_ll = Psi @ inv(R) @ V # P_parallel 
	g_ll = P_ll.T @ g	# g_Parallel
	g_NL_norm = sqrt ( abs( norm(g) ** 2 - norm(g_ll) ** 2 ) )

	sigma = 0
	phi = phi_bar_func(sigma,delta)

	if phi >= 0:
		sigma_star = 0
		tau_star = gamma
	else:
		sigma_star = solve_newton_equation_to_find_sigma(delta)
		tau_star = gamma + sigma_star

	p_star = - 1 / tau_star * \
			( g - Psi @ inv( tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )

	return p_star

def eval_reduction_ratio(sess,g,p):
	new_f = eval_aux_loss(sess,p)
	old_f = eval_loss(sess)

	ared = old_f - new_f

	if S.size is not 0:
		p_ll = P_ll.T @ p
		p_NL_norm = sqrt ( abs( norm(p) ** 2 - norm(p_ll) ** 2 ) )
		p_T_B_p = sum( Lambda_1 * p_ll ** 2)  + gamma * p_NL_norm ** 2
		pred =  - (g @ p  + 1/2 * p_T_B_p)
	else:
		pred =  - 1/2 * g @ p
	
	rho = ared / pred
	
	return rho

def eval_y(sess):
	new_g = eval_aux_gradient_vec(sess)
	old_g = g
	new_y = new_g - old_g
	return new_y

def enqueue(Z,new_val):
	if Z.size == 0:
		Z = new_val.reshape(-1,1)
		return Z
	Z = np.concatenate( (Z,new_val.reshape(-1,1)), axis=1)
	return Z
		
def dequeue(Z):
	return np.delete(Z, obj=0, axis=1)

def update_S_Y(new_s_val,new_y_val):
	global S
	global Y

	Stmp = S
	Ytmp = Y

	num_columns_S = Stmp.shape[1]
	num_columns_Y = Stmp.shape[1]
	assert num_columns_S is num_columns_Y, "dimention of S and Y doesn't match"
	if num_columns_S < m:
		Stmp = enqueue(Stmp,new_s_val)
		Ytmp = enqueue(Ytmp,new_y_val)
	else:
		Stmp = dequeue(Stmp)
		Stmp = enqueue(Stmp,new_s_val)
		Ytmp = dequeue(Ytmp)
		Ytmp = enqueue(Ytmp,new_y_val)

	S = Stmp
	Y = Ytmp
	return 


def dict_of_weight_matrices_to_single_linear_vec(x_dict):
	x_vec = np.array([])
	for key in sorted(w_tf.keys()):
		matrix = x_dict[key]
		x_vec = np.append(x_vec,matrix.flatten())	
	return x_vec

def linear_vec_to_dict_of_weight_matrices(x_vec):
	x_dict = {}
	id_start = 0
	id_end   = 0
	for key in sorted(w_tf.keys()):
		id_end = id_start + n_W[key]
		vector = x_vec[id_start:id_end]
		matrix = vector.reshape(dim_w[key])
		x_dict[key] = matrix
		id_start = id_end
	return x_dict

def compute_multibatch_tensor(sess,tensor_tf,X__,y__):
	feed_dict = {}
	total = 0
	num_minibatches_here = X__.shape[0] // minibatch
	for j in range(num_minibatches_here):
		index_minibatch = j % num_minibatches_here
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X__[start_index:end_index]
		y_batch = y__[start_index:end_index]
		feed_dict.update({	x: X_batch,
							y: y_batch})

		value = sess.run(tensor_tf, feed_dict=feed_dict)
		total = total + value

	total = total * 1 / num_minibatches_here	
	return total

def compute_multibatch_gradient(sess,grad_tf,train_set,labels):
	feed_dict = {}
	gw = {}
	num_minibatches_here = train_set.shape[0] // minibatch
	for j in range(num_minibatches_here):
		index_minibatch = j % num_minibatches_here
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = train_set[start_index:end_index]
		y_batch = labels[start_index:end_index]
		feed_dict.update({	x: X_batch,
							y: y_batch})

		gw_list = sess.run(grad_tf, feed_dict=feed_dict)
		if j == 0:		
			for layer, _ in w_tf.items():
				gw[layer] = gw_list[layer][0]
		else:
			for layer, _ in w_tf.items():
				gw[layer] = gw[layer] + gw_list[layer][0]

	for layer, _ in w_tf.items():
		gw[layer] = gw[layer] * 1 / num_minibatches_here	
	return gw

def eval_gradient_vec(sess):
	"""returns gradient, here only for mode='robust-multi-batch' 
	I should modify to consider all other cases"""
	g_dict = compute_multibatch_gradient(sess,grad_w_tf,
												X_train_multi,y_train_multi)
	g_vec = dict_of_weight_matrices_to_single_linear_vec(g_dict)
	return g_vec	

def eval_accuracy(sess):
	accuracy_val = compute_multibatch_tensor(sess,accuracy,
												X_train_multi,y_train_multi)
	return accuracy_val

def eval_accuracy_test(sess):
	accuracy_val = compute_multibatch_tensor(sess,accuracy, X_test, y_test)
	return accuracy_val


def eval_accuracy_validation(sess):
	accuracy_val = compute_multibatch_tensor(sess,accuracy, 
													X_validation,y_validation)
	return accuracy_val


def eval_w_dict(sess):
	w_dict = sess.run(w_tf)
	return w_dict

def update_weights(sess,p_vec):
	w_dict = eval_w_dict(sess)
	p_dict = linear_vec_to_dict_of_weight_matrices(p_vec)
	feed_dict = {}
	for key,_ in w_tf.items():
		feed_dict.update({update_w_placeholder[key]: w_dict[key]+p_dict[key] })
	sess.run(update_w, feed_dict=feed_dict)
	return

def eval_aux_loss(sess,p_vec):
	w_dict = eval_w_dict(sess)
	p_dict = linear_vec_to_dict_of_weight_matrices(p_vec)
	feed_dict = {}
	for key,_ in w_tf.items():
		feed_dict.update({aux_w_placeholder[key]: w_dict[key]+p_dict[key] })
	sess.run(aux_w_init,feed_dict=feed_dict)
	loss_new = compute_multibatch_tensor(sess,aux_loss,
											X_train_multi,y_train_multi)
	return loss_new

def eval_loss(sess):
	loss_val = compute_multibatch_tensor(sess,loss,X_train_multi,y_train_multi)
	return loss_val

def eval_loss_test(sess):
	loss_val = compute_multibatch_tensor(sess,loss,X_test,y_test)
	return loss_val

def eval_loss_validation(sess):
	loss_val = compute_multibatch_tensor(sess,loss,X_validation,y_validation)
	return loss_val

def eval_aux_gradient_vec(sess):
	# assuming that eval_aux_loss is being called before this function call
	aux_g_dict = compute_multibatch_gradient(sess,aux_grad_w,
												X_train_multi,y_train_multi)
	aux_g_vec = dict_of_weight_matrices_to_single_linear_vec(aux_g_dict)
	return aux_g_vec	

###############################################################################
######################## TRUST REGION ALGORITHM ###############################
###############################################################################


# save training results
loss_train_results = []
loss_validation_results = []
loss_test_results = []
accuracy_train_results = []
accuracy_validation_results = []
accuracy_test_results = []
def save_print_training_results(sess):
	loss_train = eval_loss(sess)
	accuracy_train = eval_accuracy(sess)
	loss_validation = eval_loss_validation(sess)
	accuracy_validation = eval_accuracy_validation(sess)
	loss_test = eval_loss_test(sess)
	accuracy_test = eval_accuracy_test(sess)

	# saving training results
	loss_train_results.append(loss_train)
	loss_validation_results.append(loss_validation)
	loss_test_results.append(loss_test)
	accuracy_train_results.append(accuracy_train)
	accuracy_validation_results.append(accuracy_validation)
	accuracy_test_results.append(accuracy_test)

	print('LOSS     - train: {0:.4f}, validation: {1:.4f}, test: {2:.4f}' \
						.format(loss_train, loss_validation, loss_test))
	print('ACCURACY - train: {0:.4f}, validation: {1:.4f}, test: {2:.4f}' \
			.format(accuracy_train, accuracy_validation, accuracy_test))

def permutation(n,k):
	set_1 = (k%n, k%n+1)
	set_2 = (k%n+1, k%n+2)
	if k%n == n-1:
		set_2 = (0, 1)
	return set_1, set_2

def set_multi_batch(num_batch_in_data, iteration):
	"""multi batches with half size overlap"""  

	global X_train_multi
	global y_train_multi

	if use_whole_data:
		X_train_multi = X_train
		y_train_multi = y_train
		return

	set_1, set_2 = permutation(num_batch_in_data,iteration)
	overlap_batch_size = X_train.shape[0] // num_batch_in_data
	start_index_1 = set_1[0] * overlap_batch_size
	end_index_1 = set_1[1] * overlap_batch_size
	start_index_2 = set_2[0] * overlap_batch_size
	end_index_2 = set_2[1] * overlap_batch_size

	X_half_batch_1 = X_train[start_index_1:end_index_1]
	X_half_batch_2 = X_train[start_index_2:end_index_2]
	X_train_multi = np.concatenate((X_half_batch_1,X_half_batch_2))

	y_half_batch_1 = y_train[start_index_1:end_index_1]
	y_half_batch_2 = y_train[start_index_2:end_index_2]
	y_train_multi = np.concatenate((y_half_batch_1,y_half_batch_2))

	return


def lbfgs_line_search_algorithm(sess,max_num_iter=max_num_iter):
	tolerance = 1E-5

	global gamma
	global g

	k = 0
	#-------- main loop ----------
	while(True):
		print('-'*60)
		print('iteration: {}' .format(k))

		set_multi_batch(num_batch_in_data, k)
		save_print_training_results(sess)

		g = eval_gradient_vec(sess)	
		norm_g = norm(g)
		print('norm of g = {0:.4f}' .format(norm_g))
		if norm_g < tolerance:
			print('-'*60)
			print('gradient vanished')
			print('convergence necessary but not sufficient condition') 
			print('--BREAK -- the trust region loop!')
			print('-'*60)
			break

		if k >= max_num_iter:
			print('reached to the max num iteration -- BREAK')
			break	
		
		if k == 0:
			#p = backtracking_line_search(sess,g)
			p = -g
			alpha = satisfy_wolfe_condition(sess, p)
			p = alpha*p
		else:
			p = lbfgs_line_search_subproblem_solver(sess, g)

		new_loss = eval_aux_loss(sess,p) 
		# we should call this function everytime before 
		# evaluation of aux gradient			
		new_y = eval_y(sess)
		new_s = p
		update_S_Y(new_s,new_y)
		gamma = (new_y.T @ new_y) / (new_s.T @ new_y)
		print('gamma = {0:.4f}' .format(gamma))
		update_weights(sess,p)
		print('weights are updated')

		global iter_num
		iter_num = k
				
		k += 1
	return


def lbfgs_trust_region_algorithm(sess,max_num_iter=max_num_iter):
	#--------- LOOP PARAMS ------------
	delta_hat = 3 # upper bound for trust region radius
	#max_num_iter = 1000 # max bunmber of trust region iterations
	delta = np.zeros(max_num_iter+1)
	delta[0] = delta_hat * 0.75
	rho = np.zeros(max_num_iter) # true reduction / predicted reduction ratio
	eta = 1/4 * 0.9 # eta \in [0,1/4)
	new_iteration = True
	new_iteration_number = 0
	tolerance = 1E-5

	global gamma
	global g

	k = 0
	#-------- main loop ----------
	while(True):
		print('-'*60)
		print('iteration: {}' .format(k))
		
		if new_iteration:
			set_multi_batch(num_batch_in_data, new_iteration_number)
			save_print_training_results(sess)

		g = eval_gradient_vec(sess)	
		norm_g = norm(g)
		print('norm of g = {0:.4f}' .format(norm_g))
		if norm_g < tolerance:
			print('-'*60)
			print('gradient vanished')
			print('convergence necessary but not sufficient condition') 
			print('--BREAK -- the trust region loop!')
			print('-'*60)
			break

		if k >= max_num_iter:
			print('reached to the max num iteration -- BREAK')
			break	
		
		if new_iteration_number == 0:
			p = backtracking_line_search(sess,g)
			
			# we should call this function everytime before 
			# evaluation of aux gradient
			new_loss = eval_aux_loss(sess,p) 
			new_y = eval_y(sess)
			new_s = p
			update_S_Y(new_s,new_y)
			gamma = (new_y.T @ new_y) / (new_s.T @ new_y)
			print('initial gamma = {0:.4f}' .format(gamma))
			new_iteration = True
			new_iteration_number += 1
			update_weights(sess,p)
			print('weights are updated')
			continue
		
		p = lbfgs_trust_region_subproblem_solver(delta[k], g)
		
		rho[k] = eval_reduction_ratio(sess, g, p)
		if rho[k] < 1/4:
			delta[k+1] = 1/4 * delta[k]
			print('shrinking trust region radius')
		else:
			if rho[k] > 3/4 and isclose( norm(p), delta[k] ):
				delta[k+1] = min(2*delta[k], delta_hat)
				print('expanding trust region radius')
			else:
				delta[k+1] = delta[k]

		if rho[k] > eta:
			new_y = eval_y(sess)
			new_s = p
			update_S_Y(new_s,new_y)
			gamma = (new_y.T @ new_y) / (new_s.T @ new_y)
			print('gamma = {0:.4f}' .format(gamma))
			if gamma < 0 or isclose(gamma,0):
				print('WARNING! -- gamma is not stable')
			new_iteration = True
			new_iteration_number += 1

			update_weights(sess,p)
			print('weights are updated')
		else:
			new_iteration = False
			print('-'*30)
			print('No update in this iteration')

		global iter_num
		iter_num = k

		k += 1
	return

start = time.time()

with tf.Session() as sess:
	sess.run(init)

	if method == 'trust-region':
		lbfgs_trust_region_algorithm(sess)
	elif method == 'line-search':
		lbfgs_line_search_algorithm(sess)
	else:
		print('Error! No proper method is defined')

end = time.time()

loop_time = end - start
each_iteration_avg_time = loop_time / (iter_num+1)


import pickle

result_file_path = './results/results_experiment_FEB_23_' + str(method) + '_m_' \
							+ str(m) + '_n_' + str(num_batch_in_data) + '.pkl'
if use_whole_data:
	result_file_path = './results/results_experiment_FEB_23_' + str(method) + '_m_' \
							+ str(m) + '_n_2' + '.pkl'
# Saving the objects:
with open(result_file_path, 'wb') as f: 
	pickle.dump([loss_train_results, loss_validation_results, 
													loss_test_results], f)
	pickle.dump([accuracy_train_results, accuracy_validation_results, 
													accuracy_test_results], f)
	pickle.dump([loop_time, each_iteration_avg_time], f)

# import pickle
# result_file_path = './results/results_experiment_' + str(method) + '_m_' \
# 							+ str(m) + '_n_' + str(num_batch_in_data) + '.pkl'
# with open(result_file_path,'rb') as f:  # Python 3: open(..., 'rb')
# 	loss_train_results, loss_validation_results, loss_test_results = \
# 																pickle.load(f)
# 	accuracy_train_results,accuracy_validation_results, \
# 										accuracy_test_results = pickle.load(f)
# 	loop_time, each_iteration_avg_time = pickle.load(f)

