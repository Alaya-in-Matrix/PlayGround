#include "time.h"
#include "stdio.h"
#include<iostream>
#include<vector>
#include<string>
using namespace std;
#define WEEKDAY 7
#define START_YEAR 1970
#define START_WEEKDAY 4

class Time {
    public:
	Time(int n = 0) {
	    int lt = time(NULL);
	    lt += n*3600;
	    num_day = lt/(3600*24);
	    week = num_day%WEEKDAY + START_WEEKDAY;
	    hour = (lt - num_day*3600*24)/3600;
	    min = (lt - num_day*3600*24 - hour*3600)/60;
	    sec = lt - num_day*3600*24 - hour*3600 - min*60;
	    get_date();
	}

	bool leapYear(int n) {
	    return (n%400 == 0) || (n%100 && n%4 == 0);
	}

	void get_date() {
	    int i = START_YEAR;
	    while(1) {
		int temp = leapYear(i) ? 366 : 365;
		if(num_day >= temp)
		    num_day -= temp;
		else {
		    year = i;
		    break;
		}
		i++;
	    }
	    i = 0;
	    while(1) {
		int temp = (i == 1 && leapYear(year)) ? days[i]+1 : days[i];
		if(num_day > temp)
		    num_day -= temp;
		else {
		    day = num_day+1;
		    month = i;
		    break;
		}
		i++;
	    }
	}

	string int_to_string(int n) {
	    char first = '0'+n%10;
	    char second = '0'+n/10;
	    string res = "";
	    return res + second + first;
	}

	void show() {
	    cout << weekdays[week].substr(0,3) << " ";
	    cout << months[month].substr(0,3) << " ";
	    cout << day << " ";
	    cout << int_to_string(hour) << ":";
	    cout << int_to_string(min) << ":";
	    cout << int_to_string(sec) << " ";
	    cout << year << endl;
	}

    private:
	vector<string> weekdays = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
	vector<string> months = {"Janurary","February","March","April","May","June","July","August","September","October","November","December"};
	vector<int> days = {31,28,31,30,31,30,31,31,30,31,30,31};
	int sec, min, hour, day, week, month, year;
	int num_day;
};

int main(int argc, char* argv[]) {
    int time_region = 0;
    if(argc > 1) {
	for(auto i = argv[1]+1;*i;i++)
	    time_region = time_region*10 + (*i-'0');
	time_region *= (*argv[1] < 44) ? 1 : -1;
    }
    Time t(time_region);
    t.show();
    return 0;
}
