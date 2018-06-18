#include<vector>
#include<iostream>
using namespace std;

void show2D(vector<vector<int>>& l) {
    for(int i = 0;i < l.size();i++) {
	for(int j = 0;j < l[0].size();j++)
	    cout << l[i][j] << "\t";
	cout << endl;
    }
}


void TBT(vector<vector<int>>& X, vector<vector<int>>& TR, int l) {
    int n = X.size(), m = X[0].size();
    int B = 1<<l - 1;
    for(int i = 0;i < n;i++) {
	for(int j = 0;j < m;j++) {
	    TR[i][j] = X[i][j] & B;
	    X[i][j] = X[i][j] >> l;
	}
    }     
}

int main() {
    int l = 1;
    vector<vector<int>> X = {{163,160,193,204,144,157,152,150},\
	{170,165,199,195,147,146,157,141},\
	{160,156,205,197,149,149,157,145},\
	{141,163,212,183,138,171,159,146},\
	{156,178,210,174,154,170,152,153},\
	{165,174,208,174,161,177,145,147},\
	{165,181,209,160,164,166,146,163},\
	{168,183,211,150,154,147,155,164}};
    vector<vector<int>> TR = vector<vector<int>>(X.size(), vector<int>(X[0].size(), 0));
    //show original X
    cout << "Original X:" << endl;
    show2D(X);
    cout << endl;
    // TBT
    TBT(X, TR, l);
    // show truncated X
    cout << "truncated X:" << endl;
    show2D(X);
    cout << endl;
    
    // show TR
    cout << "TR:" << endl;
    show2D(TR);
    cout << endl;

}
