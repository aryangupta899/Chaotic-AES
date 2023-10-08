#include <bits/stdc++.h>
#include <iostream>
#include <bitset>
#include <string>
#include <sys/time.h>
#include <iomanip>
#include <ios>
using namespace std;
#define N 100000


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
    float b = 0.1;
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

        if(i>200)
        {
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
    vector <char> dth={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    unordered_map <char, int> mp;
    mp['0'] = 0;
    mp['1'] = 1;
    mp['2'] = 2;
    mp['3'] = 3;
    mp['4'] = 4;
    mp['5'] = 5;
    mp['6'] = 6;
    mp['7'] = 7;
    mp['8'] = 8;
    mp['9'] = 9;
    mp['a'] = 10;
    mp['b'] = 11;
    mp['c'] = 12;
    mp['d'] = 13;
    mp['e'] = 14;
    mp['f'] = 15;
    for(int i=0; i<16; i++)
    {
        for(int j=0; j<16; j++)
        {
            stringstream ss;
            int number;
            ss <<hex<<S_Box[i][j].to_ulong();
            string s=ss.str();
            int a;
            int b;
            if(s.size()==1)
            {
                a=0;
                b=mp[s[0]];
            }
            else
            {
                a=mp[s[0]];
                b=mp[s[1]];
            }

            string str="";
            str.push_back(dth[i]);
            str.push_back(dth[j]);
            int intValue=stoi(str, nullptr, 16);
            bitset<8> bitsetFromHex(intValue);

            Inv_S_Box[a][b]=bitsetFromHex;
        }
    }
}

void Print_S_Box()
{
        for(int i=0; i<16; i++)
    {
        for(int j=0; j<16; j++)
        {
            stringstream ss;
            int number;
            ss <<hex<<S_Box[i][j].to_ulong();
            string s=ss.str();
            cout<<s<<" ";
        }
        cout<<endl;
    }

    for(int i=0; i<16; i++)
    {
        for(int j=0; j<16; j++)
        {
            stringstream ss;
            int number;
            ss <<hex<<Inv_S_Box[i][j].to_ulong();
            string s=ss.str();
            cout<<s<<" ";
        }
        cout<<endl;
    }
}

word Rcon[10] = {0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000,
                 0x20000000, 0x40000000, 0x80000000, 0x1b000000, 0x36000000};

/**********************************************************************/
/*                                                                    */
/*                              AES Algorithmic Implementation*/
/*                                                                    */
/**********************************************************************/

/******************************Here is the encrypted transformation function ****************************************************/
/**
 *  S Box Conversion - The first four bits are line numbers and the last four bits are column numbers
 */
void SubBytes(byte mtx[4 * 4])
{
    for (int i = 0; i < 16; ++i)
    {
        int row = mtx[i][7] * 8 + mtx[i][6] * 4 + mtx[i][5] * 2 + mtx[i][4];
        int col = mtx[i][3] * 8 + mtx[i][2] * 4 + mtx[i][1] * 2 + mtx[i][0];
        mtx[i] = S_Box[row][col];
    }
}

/**
 *  Line Transform - Byte Cyclic Shift
 */
void ShiftRows(byte mtx[4 * 4])
{
    // The second line circle moves one bit to the left
    byte temp = mtx[4];
    for (int i = 0; i < 3; ++i)
        mtx[i + 4] = mtx[i + 5];
    mtx[7] = temp;
    // The third line circle moves two places to the left
    for (int i = 0; i < 2; ++i)
    {
        temp = mtx[i + 8];
        mtx[i + 8] = mtx[i + 10];
        mtx[i + 10] = temp;
    }
    // The fourth line moves three left circles
    temp = mtx[15];
    for (int i = 3; i > 0; --i)
        mtx[i + 12] = mtx[i + 11];
    mtx[12] = temp;
}

/**
 *  Multiplication over Finite Fields GF(2^8)
 */
byte GFMul(byte a, byte b)
{
    byte p = 0;
    byte hi_bit_set;
    for (int counter = 0; counter < 8; counter++)
    {
        if ((b & byte(1)) != 0)
        {
            p ^= a;
        }
        hi_bit_set = (byte)(a & byte(0x80));
        a <<= 1;
        if (hi_bit_set != 0)
        {
            a ^= 0x1b; /* x^8 + x^4 + x^3 + x + 1 */
        }
        b >>= 1;
    }
    return p;
}

/**
 *  Column transformation
 */
void MixColumns(byte mtx[4 * 4])
{
    byte arr[4];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
            arr[j] = mtx[i + j * 4];

        mtx[i] = GFMul(0x02, arr[0]) ^ GFMul(0x03, arr[1]) ^ arr[2] ^ arr[3];
        mtx[i + 4] = arr[0] ^ GFMul(0x02, arr[1]) ^ GFMul(0x03, arr[2]) ^ arr[3];
        mtx[i + 8] = arr[0] ^ arr[1] ^ GFMul(0x02, arr[2]) ^ GFMul(0x03, arr[3]);
        mtx[i + 12] = GFMul(0x03, arr[0]) ^ arr[1] ^ arr[2] ^ GFMul(0x02, arr[3]);
    }
}

/**
 *  Round Key Plus Transform - XOR each column with the extended key
 */
void AddRoundKey(byte mtx[4 * 4], word k[4])
{
    for (int i = 0; i < 4; ++i)
    {
        word k1 = k[i] >> 24;
        word k2 = (k[i] << 8) >> 24;
        word k3 = (k[i] << 16) >> 24;
        word k4 = (k[i] << 24) >> 24;

        mtx[i] = mtx[i] ^ byte(k1.to_ulong());
        mtx[i + 4] = mtx[i + 4] ^ byte(k2.to_ulong());
        mtx[i + 8] = mtx[i + 8] ^ byte(k3.to_ulong());
        mtx[i + 12] = mtx[i + 12] ^ byte(k4.to_ulong());
    }
}

/**************************Here is the decrypted inverse transform function *******************************************************/
/**
 *  Inverse S-box transformation
 */
void InvSubBytes(byte mtx[4 * 4])
{
    for (int i = 0; i < 16; ++i)
    {
        int row = mtx[i][7] * 8 + mtx[i][6] * 4 + mtx[i][5] * 2 + mtx[i][4];
        int col = mtx[i][3] * 8 + mtx[i][2] * 4 + mtx[i][1] * 2 + mtx[i][0];
        mtx[i] = Inv_S_Box[row][col];
    }
}

/**
 *  Reverse Transform - Cyclic Right Shift in Bytes
 */
void InvShiftRows(byte mtx[4 * 4])
{
    // The second line circle moves one bit to the right
    byte temp = mtx[7];
    for (int i = 3; i > 0; --i)
        mtx[i + 4] = mtx[i + 3];
    mtx[4] = temp;
    // The third line circle moves two to the right
    for (int i = 0; i < 2; ++i)
    {
        temp = mtx[i + 8];
        mtx[i + 8] = mtx[i + 10];
        mtx[i + 10] = temp;
    }
    // Fourth line circle moves three to the right
    temp = mtx[12];
    for (int i = 0; i < 3; ++i)
        mtx[i + 12] = mtx[i + 13];
    mtx[15] = temp;
}

void InvMixColumns(byte mtx[4 * 4])
{
    byte arr[4];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
            arr[j] = mtx[i + j * 4];

        mtx[i] = GFMul(0x0e, arr[0]) ^ GFMul(0x0b, arr[1]) ^ GFMul(0x0d, arr[2]) ^ GFMul(0x09, arr[3]);
        mtx[i + 4] = GFMul(0x09, arr[0]) ^ GFMul(0x0e, arr[1]) ^ GFMul(0x0b, arr[2]) ^ GFMul(0x0d, arr[3]);
        mtx[i + 8] = GFMul(0x0d, arr[0]) ^ GFMul(0x09, arr[1]) ^ GFMul(0x0e, arr[2]) ^ GFMul(0x0b, arr[3]);
        mtx[i + 12] = GFMul(0x0b, arr[0]) ^ GFMul(0x0d, arr[1]) ^ GFMul(0x09, arr[2]) ^ GFMul(0x0e, arr[3]);
    }
}

/******************************Following is the key extension section ***************************************************************/
/**
 * Convert four byte s to one word.
 */
word Word(byte &k1, byte &k2, byte &k3, byte &k4)
{
    word result(0x00000000);
    word temp;
    temp = k1.to_ulong(); // K1
    temp <<= 24;
    result |= temp;
    temp = k2.to_ulong(); // K2
    temp <<= 16;
    result |= temp;
    temp = k3.to_ulong(); // K3
    temp <<= 8;
    result |= temp;
    temp = k4.to_ulong(); // K4
    result |= temp;
    return result;
}

/**
 *  Cyclic left shift by byte
 *  That is to say, [a0, a1, a2, a3] becomes [a1, a2, a3, a0]
 */
word RotWord(word &rw)
{
    word high = rw << 8;
    word low = rw >> 24;
    return high | low;
}

/**
 *  S-box transformation for each byte in input word
 */
word SubWord(word sw)
{
    word temp;
    for (int i = 0; i < 32; i += 8)
    {
        int row = sw[i + 7] * 8 + sw[i + 6] * 4 + sw[i + 5] * 2 + sw[i + 4];
        int col = sw[i + 3] * 8 + sw[i + 2] * 4 + sw[i + 1] * 2 + sw[i];
        byte val = S_Box[row][col];
        for (int j = 0; j < 8; ++j)
            temp[i + j] = val[j];
    }
    return temp;
}

/**
 *  Key Extension Function - Extended 128-bit key to w[4*(Nr+1)]
 */
void KeyExpansion(byte key[4 * Nk], word w[4 * (Nr + 1)])
{
    word temp;
    int i = 0;
    // The first four of w [] are input key s
    while (i < Nk)
    {
        w[i] = Word(key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]);
        ++i;
    }

    i = Nk;

    while (i < 4 * (Nr + 1))
    {
        temp = w[i - 1]; // Record the previous word
        if (i % Nk == 0)
            w[i] = w[i - Nk] ^ SubWord(RotWord(temp)) ^ Rcon[i / Nk - 1];
        else
            w[i] = w[i - Nk] ^ temp;
        ++i;
    }
}

/******************************Here are the encryption and decryption functions ********************************************************************/
/**
 *  encryption
 */
void encrypt(byte in[4 * 4], word w[4 * (Nr + 1)])
{
    word key[4];
    for (int i = 0; i < 4; ++i)
        key[i] = w[i];
    AddRoundKey(in, key);

    for (int round = 1; round < Nr; ++round)
    {
        SubBytes(in);
        ShiftRows(in);
        MixColumns(in);
        for (int i = 0; i < 4; ++i)
            key[i] = w[4 * round + i];
        AddRoundKey(in, key);
    }

    SubBytes(in);
    ShiftRows(in);
    for (int i = 0; i < 4; ++i)
        key[i] = w[4 * Nr + i];
    AddRoundKey(in, key);
}

/**
 *  Decrypt
 */
void decrypt(byte in[4 * 4], word w[4 * (Nr + 1)])
{
    word key[4];
    for (int i = 0; i < 4; ++i)
        key[i] = w[4 * Nr + i];
    AddRoundKey(in, key);

    for (int round = Nr - 1; round > 0; --round)
    {
        InvShiftRows(in);
        InvSubBytes(in);
        for (int i = 0; i < 4; ++i)
            key[i] = w[4 * round + i];
        AddRoundKey(in, key);
        InvMixColumns(in);
    }

    InvShiftRows(in);
    InvSubBytes(in);
    for (int i = 0; i < 4; ++i)
        key[i] = w[i];
    AddRoundKey(in, key);
}

/**********************************************************************/
/*                                                                    */
/*                              Testing*/
/*                                                                    */
/**********************************************************************/


void aes(byte plain[16])
{
    struct timeval start, end;
    // start timer.
    gettimeofday(&start, NULL);
    // unsync the I/O of C and C++.
    ios_base::sync_with_stdio(false);
    byte key[16] = {0x2b, 0x7e, 0x15, 0x16,
                    0x28, 0xae, 0xd2, 0xa6,
                    0xab, 0xf7, 0x15, 0x88,
                    0x09, 0xcf, 0x4f, 0x3c};


    word w[4 * (Nr + 1)];
    KeyExpansion(key, w);

    // Encryption, output ciphertext
    encrypt(plain, w);
}

int main()
{
    Chaotic_SBox();
    Chaotic_Inv_S_Box();

    // Define an array of std::bitset<8> values
    byte plain[16] = {0x32, 0x88, 0x31, 0xe0,
                      0x43, 0x5a, 0x31, 0x37,
                      0xf6, 0x30, 0x98, 0x07,
                      0xa8, 0x8d, 0xa2, 0x34};

    byte referenceOutput[16];
    memcpy(referenceOutput, plain, sizeof(plain));
    aes(referenceOutput);

    int totalChanges = 0;

    // Iterate through each bit position (0 to 7)
    for (size_t bitPos = 0; bitPos < 8; ++bitPos)
    {
        int bitChanges = 0;

        // Iterate through each element in the plain array
        for (size_t i = 0; i < 16; ++i)
        {
            // Create a modified input by flipping the bit at bitPos
            byte temp[16];
            memcpy(temp, plain, sizeof(plain));
            temp[i][bitPos] = !temp[i][bitPos];

            aes(temp);

            // Calculate the Hamming distance between reference and modified outputs
            int changes = (referenceOutput[i] ^ temp[i]).count();
            bitChanges += changes;

            temp[i][bitPos] = !temp[i][bitPos];
        }

        totalChanges += bitChanges;

        cout << "Bit " << bitPos << ": " << bitChanges << " bits change." << endl;
    }

    double avalancheEffect = static_cast<double>(totalChanges) / (8*8* 16);
    cout << "Avalanche Effect: " << avalancheEffect << endl;

    return 0;
}
