/**************************************************

This is a neural network with one hidden layer.

You should set these parameters apporopriately

NUM_INPUT : number of inputs
NUM_HIDDEN : number of nodes in the hidden layer
NUM_OUTPUT : number of outputs
NUM_TRAINING_DATA : number of data for training
NUM_TEST_DATA : number of data for test
MAX_EPOCH : number of iterations for learning
LEARNING_RATE : learning rate (Eta)

These variables are for training data:
double training_point[NUM_TRAINING_DATA][NUM_INPUT] : inputs of training data
double training_target[NUM_TRAINING_DATA][NUM_OUTPUT] : outputs of training data

These variables are for test data:
double test_point[NUM_TEST_DATA][NUM_INPUT]	: inputs for test
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define SIGMOID(x) (1./(1+exp(-(x))))



#define NUM_INPUT	7
#define	NUM_HIDDEN	3
#define	NUM_OUTPUT	7
#define	NUM_TRAINING_DATA	452   // rating 개수 (10만개)
#define	NUM_TEST_DATA	2 //??

#define	MAX_EPOCH	100000
#define	LEARNING_RATE	0.05


double **training_point;

double **training_target;
double **test_point;



int InitWeight(double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//weight initialization
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			weight_kj[k][j] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			weight_ji[j][i] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (k = 0; k < NUM_OUTPUT; k++)
		bias_k[k] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (j = 0; j < NUM_HIDDEN; j++)
		bias_j[j] = 1.0 * (rand() % 1000 - 500) / 5000;

	return 0;
}

int ResetDelta(double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//weight initialization
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			delta_kj[k][j] = 0;

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			delta_ji[j][i] = 0;

	for (k = 0; k < NUM_OUTPUT; k++)
		delta_bias_k[k] = 0;

	for (j = 0; j < NUM_HIDDEN; j++)
		delta_bias_j[j] = 0;

	return 0;
}

// generate outputs on the output nodes
int Forward(double training_point[NUM_INPUT],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN],
	double hidden[NUM_HIDDEN], double output[NUM_OUTPUT])

{
	int i, j, k;
	double net_j, net_k;

	//evaluate the output of hidden nodes
	for (j = 0; j < NUM_HIDDEN; j++)
	{
		net_j = 0;
		for (i = 0; i < NUM_INPUT; i++)
			net_j += weight_ji[j][i] * training_point[i];
		net_j += bias_j[j];
		hidden[j] = SIGMOID(net_j);
	}

	//evaluate the output of output nodes
	for (k = 0; k < NUM_OUTPUT; k++)
	{
		net_k = 0;
		for (j = 0; j < NUM_HIDDEN; j++)
			net_k += weight_kj[k][j] * hidden[j];
		net_k += bias_k[k];

		output[k] = SIGMOID(net_k);
	}

	return 0;
}

int Backward(double training_point[NUM_INPUT], double training_target[NUM_OUTPUT],
	double hidden[NUM_HIDDEN], double output[NUM_OUTPUT],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN],
	double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN])

{
	int i, j, k;

	//evaluate delta_kj
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			delta_kj[k][j] += -output[k] * (1 - output[k])*(training_target[k] - output[k])*hidden[j];

	for (k = 0; k < NUM_OUTPUT; k++)
		delta_bias_k[k] += -output[k] * (1 - output[k])*(training_target[k] - output[k]);

	//evaluate delta_ji
	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
		{
			double delta_k = 0;
			for (k = 0; k < NUM_OUTPUT; k++)
				delta_k += -output[k] * (1 - output[k])*(training_target[k] - output[k])*weight_kj[k][j];
			delta_ji[j][i] += delta_k*hidden[j] * (1 - hidden[j])*training_point[i];
		}

	for (j = 0; j < NUM_HIDDEN; j++)
	{
		double delta_k = 0;
		for (k = 0; k < NUM_OUTPUT; k++)
			delta_k += -output[k] * (1 - output[k])*(training_target[k] - output[k])*weight_kj[k][j];
		delta_bias_j[j] += delta_k*hidden[j] * (1 - hidden[j]);
	}

	return 0;
}

int UpdateWeights(double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//update weights
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			weight_kj[k][j] -= LEARNING_RATE*delta_kj[k][j];

	for (k = 0; k < NUM_OUTPUT; k++)
		bias_k[k] -= LEARNING_RATE*delta_bias_k[k];

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			weight_ji[j][i] -= LEARNING_RATE*delta_ji[j][i];

	for (j = 0; j < NUM_HIDDEN; j++)
		bias_j[j] -= LEARNING_RATE*delta_bias_j[j];

	return 0;
}

int PrintWeight(double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//print weights
	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			printf("%f ", weight_ji[j][i]);

	for (j = 0; j < NUM_HIDDEN; j++)
		printf("%f ", bias_j[j]);

	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			printf("%f ", weight_kj[k][j]);

	for (k = 0; k < NUM_OUTPUT; k++)
		printf("%f ", bias_k[k]);

	printf("\n");

	return 0;
}

/*
hidden[j] : output of node j at hidden layer
output[k] : output of node k at output layer
weight_kj[k][j] : weight between node j at hidden layer and node k at output layer
bias_k[k] : weight between bias (the default input, 1) and node k at output layer
weight_ji[j][i] : weight between input i and node j at hidden layer
bias_j[j] : weight between (the default input, 1) and node j at hidden layer

delta_kj[k][j] : delta for weight_kj[k][j]
delta_ji[j][i] : delta for weight_ji[j][i]
delta_bias_k[k] : delta for bias_k[k]
delta_bias_j[j] : delta for bias_j[j]

error : the summation of error
*/

int main()
{
	double hidden[NUM_HIDDEN], output[NUM_OUTPUT];
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], weight_ji[NUM_HIDDEN][NUM_INPUT];
	double bias_k[NUM_OUTPUT], bias_j[NUM_HIDDEN];
	double delta_kj[NUM_OUTPUT][NUM_HIDDEN], delta_ji[NUM_HIDDEN][NUM_INPUT];
	double delta_bias_k[NUM_OUTPUT], delta_bias_j[NUM_HIDDEN];
	double error;
	int count, I_ID=1;
	int i, k, p;

	FILE *item_user_info, *tr_user_info, *tr_test, *prediction_AE, *tr_base;
	int u_id, i_id, age, rating;
	char gender;
	double **item_user, **tr_test_, **tr_u_info;
	int num_of_rating, num_of_test;
	double MAE, abs, total_MAE, Prediction;
	int** UI;
	double user_avg[943];
	int cnt, sum;
	int integer;
	double minor;
	int RATING, USER, ITEM, TEST;
	
	/////////////////////////training point & target 동적할당!!  /////////////////////////////
	/*
	training_point = (double**)malloc(sizeof(double*)*NUM_TRAINING_DATA);
	for (count = 0; count < NUM_TRAINING_DATA; count++)
		training_point[count] = (double*)malloc(sizeof(double)*(NUM_INPUT));

	training_target = (double**)malloc(sizeof(double*)*NUM_TRAINING_DATA);
	for (count = 0; count < NUM_TRAINING_DATA; count++)
		training_target[count] = (double*)malloc(sizeof(double)*(NUM_OUTPUT));
		*/
	fopen_s(&item_user_info, "item_user_info.data", "rt");
	fopen_s(&tr_user_info, "tr_user_info.data", "rt");
	fopen_s(&tr_test, "tr_test.data", "rt");
	fopen_s(&tr_base, "tr_base.data", "rt");
	fopen_s(&prediction_AE, "Prediction by AE_test1.data", "wt");

	printf("Fix factors\n");
	printf("number of rating : ");
	scanf("%d", &RATING);
	printf("number of user : ");
	scanf("%d", &USER);
	printf("number of item : ");
	scanf("%d", &ITEM);
	printf("number of empty rating : ");
	scanf("%d", &TEST);

	UI = (int**)malloc(sizeof(int*)*USER);      //동적할당!! user-item matrix 생성
	for (i = 0; i < USER; i++) {
		UI[i] = (int*)malloc(sizeof(int)*(ITEM));
	}

	item_user = (double**)malloc(sizeof(double*) * RATING);
	for (i = 0; i < RATING; i++)
		item_user[i] = (double*)malloc(sizeof(double) * 8);

	tr_test_ = (double**)malloc(sizeof(double*) * TEST);
	for (i = 0; i < TEST; i++)
		tr_test_[i] = (double*)malloc(sizeof(double) * 7);

	tr_u_info = (double**)malloc(sizeof(double*) * USER);
	for (i = 0; i < USER; i++)
		tr_u_info[i] = (double*)malloc(sizeof(double) * 3);
	puts("set of matricies generating...");
	//////////////////////////////////////tr_user_info matrix 생성////////////////////////////
	for (i = 0; ; i++) {
		if (fscanf_s(tr_user_info, "%d %d %c", &u_id, &age, &gender, sizeof(gender)) == EOF)
			break;
		tr_u_info[i][0] = u_id;
		tr_u_info[i][1] = (double)age / 100;
		if (gender == 'M')
			tr_u_info[i][2] = 0;
		else
			tr_u_info[i][2] = 1;
	}
	/////////////////////////////////////////user-item matrix 생성!!/////////////////////////////////////
	for (i = 0; i < USER; i++) {
		for (k = 0; k < ITEM; k++) {
			UI[i][k] = 0;                // 0은 빈칸이라는 뜻! 빈칸으로 초기화
		}
	}

	while (1) {
		if (fscanf_s(tr_base, "%d %d %d", &u_id, &i_id, &rating) == EOF)
			break;
		UI[u_id - 1][i_id - 1] = rating;

	}
	for (i = 0; i < USER; i++) {    // 평균 rating 계산
		cnt = 0;
		sum = 0;
		for (k = 0; k < ITEM; k++) {
			if (UI[i][k]>0) {
				cnt++;
				sum += UI[i][k];
			}
		}
		user_avg[i] = (double)sum / cnt;
	}
	
	////////////////////////////tr_test matirx 생성/////////////////////////////////
	
	for (i = 0; ; i++) {
		if (fscanf_s(tr_test, "%d %d %d", &u_id, &i_id, &rating) == EOF)
			break;
		tr_test_[i][0] = u_id;
		tr_test_[i][1] = i_id;
		if (rating == 1) {
			tr_test_[i][2] = 1;
			tr_test_[i][3] = 0;
			tr_test_[i][4] = 0;
			tr_test_[i][5] = 0;
			tr_test_[i][6] = 0;
		}
		else if (rating == 2) {
			tr_test_[i][2] = 0;
			tr_test_[i][3] = 1;
			tr_test_[i][4] = 0;
			tr_test_[i][5] = 0;
			tr_test_[i][6] = 0;
		}
		else if (rating == 3) {
			tr_test_[i][2] = 0;
			tr_test_[i][3] = 0;
			tr_test_[i][4] = 1;
			tr_test_[i][5] = 0;
			tr_test_[i][6] = 0;
		}
		else if (rating == 4) {
			tr_test_[i][2] = 0;
			tr_test_[i][3] = 0;
			tr_test_[i][4] = 0;
			tr_test_[i][5] = 1;
			tr_test_[i][6] = 0;
		}
		else if (rating == 5) {
			tr_test_[i][2] = 0;
			tr_test_[i][3] = 0;
			tr_test_[i][4] = 0;
			tr_test_[i][5] = 0;
			tr_test_[i][6] = 1;
		}
	}
	
	
	//////////////////////////////////////item_user_info matrix 생성!! ////////////////////////////////////
	for (i = 0; ; i++) {
		if (fscanf_s(item_user_info, "%d %d %c %d", &i_id, &age, &gender, sizeof(gender), &rating) == EOF)
			break;

		item_user[i][0] = (double)age / 100;
		if (gender == 'M')
			item_user[i][1] = 0;   //남자면 0
		else
			item_user[i][1] = 1;   //여자면 1
		if (rating == 1) {
			item_user[i][2] = 1;
			item_user[i][3] = 0;
			item_user[i][4] = 0;
			item_user[i][5] = 0;
			item_user[i][6] = 0;
		}
		else if (rating == 2) {
			item_user[i][2] = 0;
			item_user[i][3] = 1;
			item_user[i][4] = 0;
			item_user[i][5] = 0;
			item_user[i][6] = 0;
		}
		else if (rating == 3) {
			item_user[i][2] = 0;
			item_user[i][3] = 0;
			item_user[i][4] = 1;
			item_user[i][5] = 0;
			item_user[i][6] = 0;
		}
		else if (rating == 4) {
			item_user[i][2] = 0;
			item_user[i][3] = 0;
			item_user[i][4] = 0;
			item_user[i][5] = 1;
			item_user[i][6] = 0;
		}
		else if (rating == 5) {
			item_user[i][2] = 0;
			item_user[i][3] = 0;
			item_user[i][4] = 0;
			item_user[i][5] = 0;
			item_user[i][6] = 1;
		}

		item_user[i][7] = i_id;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////
	puts("set of matricies generated");
	
	fclose(item_user_info);
	fclose(tr_test);
	fclose(tr_user_info);
	/////////////////////////////////////각 i_id마다 rating개수 계산 후 동적할당!!///////////////////////////////////////
	for (I_ID = 1; I_ID <= ITEM; I_ID++) {
		count = 0;
		for (i = 0; i < RATING; i++) {
			if (item_user[i][7] == I_ID)
				count++;
		}
		num_of_rating = count;

		
		count = 0;
		for (i = 0; i < TEST; i++) {
			if (tr_test_[i][1] == I_ID)
				count++;
		}
		num_of_test = count;
		
		//(num_of_rating - num_of_rest)
		training_point = (double**)malloc(sizeof(double*)*num_of_rating);
		for (count = 0; count < num_of_rating; count++)
			training_point[count] = (double*)malloc(sizeof(double)*(NUM_INPUT));

		training_target = (double**)malloc(sizeof(double*)*num_of_rating);
		for (count = 0; count < num_of_rating; count++)
			training_target[count] = (double*)malloc(sizeof(double)*(NUM_OUTPUT));

		test_point = (double**)malloc(sizeof(double*)*num_of_test);
		for (count = 0; count < num_of_test; count++)
			test_point[count] = (double*)malloc(sizeof(double)*(NUM_INPUT + 1));

		////////////////////////////////training point & target 초기화!! ////////////////////////////////////////////////////
		count = 0;
		printf("%d %d\n", num_of_rating, num_of_test);

		for (i = 0; i < RATING; i++) {
			if (item_user[i][7] == I_ID) {
				for (k = 0; k < 7; k++) {
					training_point[count][k] = item_user[i][k];
					training_target[count][k] = item_user[i][k];
				}
				count++;
			}
		}
		
		count = 0;
		for (i = 0; i < TEST; i++) {
			if (tr_test_[i][1] == I_ID) {
				for (k = 0; k < 943; k++) {
					if (tr_test_[i][0] == tr_u_info[k][0]) {
						test_point[count][0] = tr_u_info[k][1]; //age
						test_point[count][1] = tr_u_info[k][2]; //gender
						/////rating///////
						integer = (int)user_avg[k];
						minor = user_avg[k] - integer;

						if (minor == 0) {
							if (user_avg[k] == 1) {
								test_point[count][2] = 1;
								test_point[count][3] = 0;
								test_point[count][4] = 0;
								test_point[count][5] = 0;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] == 2) {
								test_point[count][2] = 0;
								test_point[count][3] = 1;
								test_point[count][4] = 0;
								test_point[count][5] = 0;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] == 3) {
								test_point[count][2] = 0;
								test_point[count][3] = 0;
								test_point[count][4] = 1;
								test_point[count][5] = 0;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] == 4) {
								test_point[count][2] = 0;
								test_point[count][3] = 0;
								test_point[count][4] = 0;
								test_point[count][5] = 1;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] == 5) {
								test_point[count][2] = 0;
								test_point[count][3] = 0;
								test_point[count][4] = 0;
								test_point[count][5] = 0;
								test_point[count][6] = 1;
							}
						}
						else if (minor != 0) {
							if (user_avg[k] > 1 && user_avg[k] <2) {
								test_point[count][2] = 1-minor;
								test_point[count][3] = minor;
								test_point[count][4] = 0;
								test_point[count][5] = 0;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] > 2 && user_avg[k] <3) {
								test_point[count][2] = 0;
								test_point[count][3] = 1-minor;
								test_point[count][4] = minor;
								test_point[count][5] = 0;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] > 3 && user_avg[k] <4) {
								test_point[count][2] = 0;
								test_point[count][3] = 0;
								test_point[count][4] = 1-minor;
								test_point[count][5] = minor;
								test_point[count][6] = 0;
							}
							else if (user_avg[k] > 4 && user_avg[k] <5) {
								test_point[count][2] = 0;
								test_point[count][3] = 0;
								test_point[count][4] = 0;
								test_point[count][5] = 1-minor;
								test_point[count][6] = minor;
							}
						}
						
						test_point[count][7] = tr_test_[i][0];  //u_id 학습하지는 않는다!!
						count++;
					}
				}
			}
			
		}
		
		///////////////////////////////////////////////////////////////////////////////////////////

		srand((unsigned)time(NULL));

		InitWeight(weight_kj, weight_ji, bias_k, bias_j);

		// loop for learning
		printf("******* Training of NN (Iteration : Error) *******\n");

		for (int epoch = 0; epoch <= MAX_EPOCH; epoch++)
		{
			error = 0;

			ResetDelta(delta_kj, delta_ji, delta_bias_k, delta_bias_j);

			for (p = 0; p < num_of_rating; p++)
			{
				Forward(training_point[p], weight_kj, weight_ji, bias_k, bias_j,
					hidden, output);

				for (k = 0; k < NUM_OUTPUT; k++)
					error += (output[k] - training_target[p][k])*(output[k] - training_target[p][k]);

				Backward(training_point[p], training_target[p], hidden, output, weight_kj,
					delta_kj, delta_ji, delta_bias_k, delta_bias_j);
			}

			UpdateWeights(delta_kj, delta_ji, delta_bias_k, delta_bias_j,
				weight_kj, weight_ji, bias_k, bias_j);

			if (epoch % 1000 == 0) printf("%d: %f\n", epoch, error);
		}
		
		// testing with un-learned point 
		printf("\n");
		printf("******* Test of NN (Input ; Output of NN) *******\n");
		total_MAE = 0;
		count = 0;
		for (i = 0; i < num_of_test; i++)
		{
			Forward(test_point[i], weight_kj, weight_ji, bias_k, bias_j,
				hidden, output);
			/*
			for (k = 0; k < NUM_INPUT; k++)
				printf("%f ", test_point[i][k]);
			printf("; ");

			for (k = 0; k < NUM_OUTPUT; k++)
			{
				printf("%f ", output[k]);
			}
			printf("\n");
			*/
		
			Prediction = 1 * output[2] + 2 * output[3] + 3 * output[4] + 4 * output[5] + 5 * output[6];
			Prediction = Prediction / (output[2] + output[3] + output[4] + output[5] + output[6]);
			fprintf(prediction_AE, "%d %d %f\n", (int)test_point[i][7], I_ID, Prediction);


			MAE = 0;
			for (k = 0; k < 20000; k++) {
				if ((tr_test_[k][0] == test_point[i][7]) && (tr_test_[k][1] == I_ID)) {
					if (tr_test_[k][2] == 1)
						abs = 1;
					else if (tr_test_[k][3] == 1)
						abs = 2;
					else if (tr_test_[k][4] == 1)
						abs = 3;
					else if (tr_test_[k][5] == 1)
						abs = 4;
					else if (tr_test_[k][6] == 1)
						abs = 5;
					abs = abs - Prediction;
					if (abs < 0)
						abs = -abs;
					MAE += abs;
					count++;
				}
			}
			total_MAE += MAE;

		}

		total_MAE = total_MAE / num_of_test;
		//printf("i_id : %d, # of rating : %d %d, total_MAE = %f\n", I_ID, count, num_of_test, total_MAE);
		
		for (count = 0; count < num_of_rating; count++) {
			free(training_point[count]);
			free(training_target[count]);
		}
		free(training_point);
		free(training_target);
		for (count = 0; count < num_of_test; count++)
			free(test_point[count]);
		free(test_point);
	}
}
