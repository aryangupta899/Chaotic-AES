#include <bits/stdc++.h>
#include <iostream>
#include <bitset>
#include <string>
#include <sys/time.h>
#include <iomanip>
#include <ios>
using namespace std;
#define N 100000

// 20520562 - Nguyen Dinh Kha
// a = 0.2, b = 0.2 and c = 5.7 . "a" was chosen as the bifurcation parameter
// properties of a = 0.1, b = 0.1, and c = 14 have been more commonly used


typedef bitset<8> byte;
typedef bitset<32> word;

const int Nr = 10; // AES-128 requires 10 rounds of encryption
const int Nk = 4;  // Nk Represents the number of word s that are input keys

byte S_Box[16][16];
void Chaotic_SBox()
{
    double dt = 0.002;
    double dx, dy, dz;

    float a = 0.2;  // standard parameter value selected by Rossler
    float b = 0.2;
    float c = 5.7;

    double x = 0; // strange attractor with unidentified variables
    double y = 0;
    double z = 0;

    unordered_set <byte> st;

    for (int i = 0; i < N; i++)
    {
        // 3 coupled non-linear differential equations:
        dx = x + (-y - z) * dt;
        dy = y + (x + a * y) * dt;
        dz = z + (b + z * (x - c)) * dt;
        x = dx;
        y = dy;
        z = dz;

        int x6=(int)(x*1000000)%10;
        int y5=(int)(y*100000)%10;
        int z4=(int)(z*10000)%10;
        int num=(x6*100+y5*10+z4)%256;
        num=abs(num);
        byte b=bitset<8>(num);
        st.insert(b);
        if(st.size()==256)
            break;
    }

    int i=0;
    int j=0;

    for(auto it: st)
    {
        S_Box[i][j]=it;
        j++;
        if(j==16)
        {
            i++;
            j=0;
        }
    }
}

byte Inv_S_Box[16][16];
void Chaotic_Inv_S_Box()
{
    for(int i=0; i<16; i++)
    {
        for(int j=0; j<16; j++)
        {
            stringstream ss;
            int number;
            ss <<hex<<S_Box[i][j].to_ulong();
            string s=ss.str();
            pair <int, int> coords=mapping(s);

            cout<<s<<" ";
        }
        cout<<endl;
    }
}

int main()
{
    Chaotic_SBox();
    Chaotic_Inv_S_Box();
    return 0;
}
