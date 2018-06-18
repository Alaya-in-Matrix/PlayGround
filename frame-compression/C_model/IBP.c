#include<iostream>
#include<vector>
#include<math.h>
using namespace std;

void show2D(vector<vector<int>>& l) {
    for(int i = 0;i < l.size();i++) {
	for(int j = 0;j < l[i].size();j++)
	    cout << l[i][j] << "\t";
	cout << endl;
    }
}

int PSNR(vector<vector<int>>& X, vector<vector<int>>& Y,int l) {
    int MAX = (1<<l - 1)*(1<<l - 1);
    int MSE = 0.0;
    for(int i = 0;i < X.size();i++) 
	for(int j = 0;j < X[0].size();j++)
	    MSE += pow(X[i][j] - Y[i][j],2);
    MSE = MSE/(X.size() * X[0].size());
    return 10*log10(MAX/MSE);
}

void IBP(vector<vector<int>>& X, int intra_mode, int l, int& IP, int& BP_mode, vector<vector<int>>& BP_res, vector<vector<int>>& NP_res) {
    // eight IBP mode
    auto mode = [](int intra_mode,int r1,int r2,int r3,int r4) {
	switch(intra_mode) {
	    case 0: return r1;
	    case 1: return r3;
	    case 2: return (r1+r2)>>1;
	    case 3: return (r3+r4)>>1;
	    case 4: return (r1+r4)>>1;
	    case 5: return (r1+r3)>>1;
	    case 6: return (r1+r2)>>2 + r3>>1;
	    case 7: return (r1+r2+r3+r4)>>2;
	    default: return r1;
	}
    };
    // get X shape
    int n = X.size(), m = X[0].size();
    vector<vector<int>> domain = {{4,5,7},{4,5,7},{0,4,5},{0,2,6},{1,5,6},{1,3,4}};
    vector<int> modelist = domain[intra_mode];

    // IP
    IP = X[0][0];

    //left BP residual
    for(int i = 1;i < n;i++)
	BP_res[0][i] = X[i][0] - X[i-1][0];
    //top BP residual
    for(int i = 1;i < m;i++)
	BP_res[1][i] = X[0][i] - X[0][i-1];

    //predicts
    int p = 0.0, best_predict = 0;
    vector<vector<vector<int>>> predicts = vector<vector<vector<int>>>(modelist.size(),vector<vector<int>>(n,vector<int>(m,0)));
    int r1 = 0, r2 = 0, r3 = 0, r4 = 0;
    for(int k = 0;k < modelist.size();k++) {
	for(int i = 0;i < n;i++) {
	    for(int j = 0;j < m;j++) {
		if(i == 0 || j == 0) 
		    predicts[k][i][j] = X[i][j];
		else {
		    r1 = X[i][j-1];
		    r2 = X[i-1][j-1];
		    r3 = X[i-1][j];
		    r4 = j+2>=m?0:X[i-1][j+1];
		    predicts[k][i][j] = mode(modelist[k],r1,r2,r3,r4);
		}
	    }
	}
	int current = PSNR(X, predicts[k], l);
	if(current > p) {
	    BP_mode = modelist[k];
	    best_predict = k;
	}
    }

    //NP residual
    for(int i = 0;i < n-1;i++) 
	for(int j = 0;j < m-1;j++)
	    NP_res[i][j] = X[i+1][j+1] - predicts[best_predict][i+1][j+1];
}

int main() {
    int l = 7;
    int intra_mode = 0, IP = 0, BP_mode = 0;
    vector<vector<int>> X = {{163,160,193,204,144,157,152,150},\
	{170,165,199,195,147,146,157,141},\
	{160,156,205,197,149,149,157,145},\
	{141,163,212,183,138,171,159,146},\
	{156,178,210,174,154,170,152,153},\
	{165,174,208,174,161,177,145,147},\
	{165,181,209,160,164,166,146,163},\
	{168,183,211,150,154,147,155,164}};
    int n = X.size(), m = X[0].size();
    //BP_res: left + top
    vector<vector<int>> BP_res = {vector<int>(n-1,0), vector<int>(m-1,0)};
    vector<vector<int>> NP_res = vector<vector<int>>(n-1,vector<int>(m-1,0));
    IBP(X, intra_mode, l, IP, BP_mode, BP_res, NP_res);
    cout << "X:" << endl;
    show2D(X);
    cout << "BP residual:" << endl;
    show2D(BP_res);
    cout << "NP residual:" << endl;
    show2D(NP_res);

}


