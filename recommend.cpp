#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define NUM_ITEM	1682
#define NUM_USER	943

int abs(int num) {
	if (num < 0)
		return -num;
	else
		return num;
}

double abs_d(double num) {
	if (num < 0)
		return -num;
	else
		return num;
}

int Calculate_Num_Of_Neighbor(int target_user, int target_movie, int** UI, double** R) {  ////이웃 몇명인지 계산!
	int i, NON=0;

	for (i = 0; i < NUM_USER; i++) {
		if ((UI[i][target_movie]>0) && (i != target_user) && (R[i][target_user] > 0))  // threshold??
			NON++;
	}

	return NON;
}

double Prediction_Neighbor_25(int target_user, int target_movie, int NON, int** UI, double** R, int max_neighbor) {
	int *neighbor;
	int i, k = -1;
	int temp;
	int K;
	double Prediction, term1 = 0, term2 = 0;

	neighbor = (int*)malloc(sizeof(int)*NON);    // neighbor들의 user_id를 저장하는 배열 생성

	for (i = 0; i < NUM_USER; i++) {
		if ((UI[i][target_movie]>0) && (i != target_user) && (R[i][target_user] > 0)) {
			k++;
			neighbor[k] = i;
		}
	}

	if ((k + 1) != NON)            // error check!
		puts("NON error!");

	for (i = 0; i < NON - 1; i++) {                   //neighbor 배열을 유사도에 따라 내림차순으로 정렬!
		for (k = 0; k < NON - 1 - i; k++) {
			if (R[target_user][neighbor[k]] < R[target_user][neighbor[k + 1]]) {
				temp = neighbor[k];
				neighbor[k] = neighbor[k + 1];
				neighbor[k + 1] = temp;
			}
		}
	}

	if (NON < max_neighbor) {
		K = NON;
	}
	else
		K = max_neighbor;

	if (K == 0)
		return R[target_user][NUM_USER];
	

	for (i = 0; i < K; i++) {              //예측점수 계산!
		term1 += R[target_user][neighbor[i]]*(UI[neighbor[i]][target_movie] - R[neighbor[i]][NUM_USER]);
		term2 += R[target_user][neighbor[i]];
	}

	

	Prediction = R[target_user][NUM_USER] + term1 / term2;

	return Prediction;
}

double Proposed_Prediction_Neighbor_25(int target_user, int target_movie, int NON, int** UI, double** R, double** UI_AE, int max_neighbor) {
	int *neighbor;
	int i, k = -1;
	int temp;
	int K;
	double Prediction, term1 = 0, term2 = 0;
	double Neighbor_Saturation;              //이웃 포화도

	neighbor = (int*)malloc(sizeof(int)*NON);    // neighbor들의 user_id를 저장하는 배열 생성

	for (i = 0; i < NUM_USER; i++) {
		if ((UI[i][target_movie]>0) && (i != target_user) && (R[i][target_user] > 0)) {
			k++;
			neighbor[k] = i;
		}
	}

	if ((k + 1) != NON)            // error check!
		puts("NON error!");

	for (i = 0; i < NON - 1; i++) {                   //neighbor 배열을 유사도에 따라 내림차순으로 정렬!
		for (k = 0; k < NON - 1 - i; k++) {
			if (R[target_user][neighbor[k]] < R[target_user][neighbor[k + 1]]) {
				temp = neighbor[k];
				neighbor[k] = neighbor[k + 1];
				neighbor[k + 1] = temp;
			}
		}
	}

	if (NON < max_neighbor) {
		K = NON;
		Neighbor_Saturation = (double)NON/25;
	}
	else {
		K = max_neighbor;
		Neighbor_Saturation = 1;
	}

	if (K == 0)
		return UI_AE[target_user][target_movie]; // AE로 예측한 점수


	for (i = 0; i < K; i++) {              //예측점수 계산!
		term1 += R[target_user][neighbor[i]] * (UI[neighbor[i]][target_movie] - R[neighbor[i]][NUM_USER]);
		term2 += R[target_user][neighbor[i]];
	}



	Prediction = R[target_user][NUM_USER] + term1 / term2;

	Prediction = Neighbor_Saturation*Prediction + (1 - Neighbor_Saturation)*R[target_user][NUM_USER];   //UI_AE[target_user][target_movie];

	return Prediction;
}

int main() {
	FILE *base, *test, *AE, *test_predict;
	int **UI;
	double **UI_AE;
	double **R;
	int i, k, j;
	int u_id, i_id, rating;
	double rating_AE;
	int cnt, sum, check=0;
	double up_P, down_P1, down_P2; // pearson correlation 분자, 분모
	int NON; //Number Of Neighbor
	double Proposed_Prediction, Proposed_MAE, Proposed_RMSE;
	int max_neighbor, max_NON, that_movie;

	fopen_s(&base, "tr_base1.data", "rt");
	fopen_s(&test, "tr_test1.data", "rt");
	fopen_s(&AE, "Prediction by AE_test1.data", "rt");
	fopen_s(&test_predict, "test_predict.data", "wt");

	UI = (int**)malloc(sizeof(int*)*NUM_USER);      //동적할당!! user-item matrix 생성
	for (i = 0; i < NUM_USER; i++) {
		UI[i] = (int*)malloc(sizeof(int)*(NUM_ITEM));
	}

	UI_AE = (double**)malloc(sizeof(double*)*NUM_USER);
	for (i = 0; i < NUM_USER; i++)
		UI_AE[i] = (double*)malloc(sizeof(double)*NUM_ITEM);

	R = (double**)malloc(sizeof(double*)*NUM_USER);  // User-User Relation matrix   R[NUM_USER] : 평균 rating
	for (i = 0; i < NUM_USER; i++){
		R[i] = (double*)malloc(sizeof(double) * (NUM_USER+1));
	}

	//////////////////////////// user-item Rating matrix 생성 /////////////////////////////////////////////////////////////////
	

	puts("User-Item Matrix Genarating...");
	for (i = 0; i < NUM_USER; i++) {
		for (k = 0; k < NUM_ITEM; k++) {
			UI[i][k] = 0;                // 0은 빈칸이라는 뜻! 빈칸으로 초기화
		}
	}
	
	while (1) {
		if (fscanf_s(base, "%d %d %d", &u_id, &i_id, &rating) == EOF)
			break;
		UI[u_id-1][i_id-1] = rating;
	}

	while (1) {
		if (fscanf_s(test, "%d %d %d", &u_id, &i_id, &rating) == EOF)
			break;
		UI[u_id-1][i_id-1] = -rating;         //음수이면 테스트를 위해 빈칸을 뚫어놓았다는 뜻! 절대값은 정답 rating을 의미!
	}
	puts("User-Item Matrix Genarated");
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////// user-item Rating by AE matrix 생성 /////////////////////////////////////////////////////////////////


	puts("User-Item AE Matrix Genarating...");
	for (i = 0; i < NUM_USER; i++) {
		for (k = 0; k < NUM_ITEM; k++) {
			UI_AE[i][k] = 0;                // 0은 빈칸이라는 뜻! 빈칸으로 초기화
		}
	}

	while (1) {
		if (fscanf_s(AE, "%d %d %lf", &u_id, &i_id, &rating_AE) == EOF)
			break;
		UI_AE[u_id - 1][i_id - 1] = rating_AE;
	}
	puts("User-Item AE Matrix Genarated");
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////user-user Relation matirx 생성 /////////////////////////////////////////////////////////////////

	puts("User-User Relation Matrix Genarating...");
	for (i = 0; i < NUM_USER; i++) {    // 평균 rating 계산
		cnt = 0;
		sum = 0;
		for (k = 0; k < NUM_ITEM; k++) {
			if (UI[i][k]>0) {
				cnt++;
				sum += UI[i][k];
			}
		}
		R[i][NUM_USER] = (double)sum / cnt;
	}

	for (i = 0; i < NUM_USER; i++) {                 // pearson correlation을 계산하여 relation matirx 완성!
		for (j = 0; j < NUM_USER; j++) {
			up_P = 0;
			down_P1 = 0;
			down_P2 = 0;
			cnt = 0;
			for (k = 0; k < NUM_ITEM; k++) {
				if ((UI[i][k] > 0) && (UI[j][k] > 0)) {   // user i와 user j 모두 본 영화에 대해서!!
					cnt++;
					up_P += (abs(UI[i][k]) - R[i][NUM_USER])*(abs(UI[j][k]) - R[j][NUM_USER]);
					down_P1 += (abs(UI[i][k]) - R[i][NUM_USER])*(abs(UI[i][k]) - R[i][NUM_USER]);
					down_P2 += (abs(UI[j][k]) - R[j][NUM_USER])*(abs(UI[j][k]) - R[j][NUM_USER]);
					
				}
			}
			if (cnt == 0) {
				R[i][j] = 0;
				continue;
			}
			if (down_P1*down_P2 == 0) { ////////// ??????????????????????????????????????????????????
				R[i][j] = 0;
				continue;
			}

			R[i][j] = up_P / sqrt(down_P1*down_P2);

			

		}
	}
	puts("User-User Relation Matrix Genarated");

	///////////////////////////////////////Rating Prediction//////////////////////////////////////////////
	puts("Rating Prediction");

	cnt = 0;
	Proposed_MAE = 0;
	Proposed_RMSE = 0;
	max_neighbor =25;
	max_NON = 0;
	for (i = 0; i < NUM_USER; i++){
		for (j = 0; j < NUM_ITEM; j++) {
			if (UI[i][j] < 0) {            //target 찾는다..  이웃 25명 찾아야하는데...
				cnt++;
				NON = Calculate_Num_Of_Neighbor(i, j, UI, R);   // target user와 movie에 대한 이웃이 몇명인가 계산
				if (NON > max_NON) {
					max_NON = NON;
					that_movie = j;
				}
				if (NON == 0)
					check++;
				Proposed_Prediction = Proposed_Prediction_Neighbor_25(i, j, NON, UI, R, UI_AE, max_neighbor);

				fprintf(test_predict, "%d %d %f\n", i+1, j+1, Proposed_Prediction);
				
				Proposed_MAE += abs_d(Proposed_Prediction - abs(UI[i][j]));
				Proposed_RMSE += (Proposed_Prediction - abs(UI[i][j]))*(Proposed_Prediction - abs(UI[i][j]));
			}
		}
	}
	printf("NON is zero : %d\n", check);
	printf("MAX NON = %d & that movie %d\n", max_NON, that_movie);
	puts("complete!");

	Proposed_MAE = Proposed_MAE / cnt;
	Proposed_RMSE = sqrt(Proposed_RMSE / cnt);
	//printf("Proposed_MAE = %f,  Proposed_RMSE = %f\n", Proposed_MAE, Proposed_RMSE);

	fclose(base);
	fclose(test);
	fclose(AE);

	for (i = 0; i < NUM_USER; i++) {
		free(UI[i]);
		free(R[i]);
		free(UI_AE[i]);
	}
	free(UI);
	free(R);
	free(UI_AE);
	return 0;
}