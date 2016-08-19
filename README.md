1. explain about software
	This software is for movie recommendation.
2. how to build
	1) Upload API
		excute upload.cpp
	2) Recommend API
		excute recommend.cpp
3. instruction
	1) Upload API
		In Upload.cpp, you can input your data. Data should be contained in file that stored the code.
		
		INPUT
		-item_user_into.data : Information between item and user
		item_ID    age    gender    rating

		-tr_user_info.data : User information
		user_ID    age    gender

		-tr_base.data : user-item rating information
		user_ID    item_ID    rating

		-tr_test.data : user-item rating information that you want to know(rating is possible any integer)
		user_ID    item_ID    rating

		OUTPUT
		-Prediction_AE.data : Rating prediction result by using auto-encoder
	2) Recommend API
		In Recommend.cpp

		INPUT
		-tr_base.data
		-tr_test.data
		-Prediction_AE.data

		OUTPUT
		-test_predict.data : Rating prediction result using by proposed method. You can get prediction ratings about tr_test.data
