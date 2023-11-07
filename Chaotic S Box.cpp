#include <bits/stdc++.h>
#include <stdlib.h>
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
unsigned int sbox[256];
void Chaotic_SBox()
{
    double dt = 0.002;
    double dx, dy, dz;

    float a = 0.2;  // standard parameter value selected by Rossler
    float b = 0.2;
    float c = 5.6;

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
            int x3=(int)(x*1000)%10;
            int x4=(int)(x*10000)%10;
            int x5=(int)(x*100000)%10;
            int y3=(int)(y*1000)%10;
            int y4=(int)(y*10000)%10;
            int y5=(int)(y*100000)%10;
            int num=(y3*1000+y4*100+x3*10+x4)%256;
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
        unsigned int int_value = it.to_ulong();
        sbox[i*10+j]=int_value;
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


/**********************************************************************/
/*                                                                    */
/*                      SAC and Bic vagera ka code
/*                                                                    */
/**********************************************************************/

#define two_power(n) (1u << (n))

#define array_size(a) (sizeof(a) / sizeof(*a))


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

unsigned int array_max(unsigned int *arr, unsigned int length) {
    unsigned int i;
    unsigned int max = arr[0];
    for (i = 1; i < length; ++i) {
        if (max < arr[i]) {
            max = arr[i];
        }
    }
    return max;
}

unsigned int log_2(unsigned int v) {
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious

    unsigned int r = 0; // r will be lg(v)
    while (v >>= 1) // unroll for more speed...
    {
        r++;
    }
    return r;
}

unsigned int nbits(unsigned int i) {
    if (i == 0) return 0;
    return log_2(i) + 1;
}

unsigned int parity(unsigned int v) {
    // https://graphics.stanford.edu/~seander/bithacks.html#ParityParallel

    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v &= 0xf;
    return (0x6996 >> v) & 1u;
}

unsigned int hamming_weight(unsigned int v) {
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
    return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; // count
}

double correlation(unsigned int *x, unsigned int *y, unsigned int n) {

    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    int i;

    for(i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];

        sx += xi;
        sy += yi;
        sxx += xi * xi;
        syy += yi * yi;
        sxy += xi * yi;
    }

    // covariation
    double cov = sxy / n - sx * sy / n / n;
    if (cov == 0) return 0.0;

    // standard error of x
    double sigmax = sqrt(sxx / n -  sx * sx / n / n);
    // standard error of y
    double sigmay = sqrt(syy / n -  sy * sy / n / n);

    // correlation is just a normalized covariation
    return cov / sigmax / sigmay;
}

// matrix allocate and print

unsigned int **alloc_uint_matrix(unsigned int m, unsigned int n) {
    unsigned int **mat;
    mat = new unsigned int *[m];
    for (unsigned int i = 0; i < m; ++i) {
        mat[i] = new unsigned int[n]();
    }
    return mat;
}

void free_uint_matrix(unsigned int **mat, unsigned int m) {
    for (unsigned int i = 0; i < m; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}

void print_uint_matrix(unsigned int **mat, unsigned int m, unsigned int n) {

    unsigned int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            printf("%2d ", mat[i][j]);
        }
        printf("\n");
    }

}

float **alloc_float_matrix(unsigned int m, unsigned int n) {
    float **mat;
    mat = new float *[m];
    for (unsigned int i = 0; i < m; ++i) {
        mat[i] = new float[n]();
    }
    return mat;
}

void free_float_matrix(float **matrix, unsigned int rows) {
    for (unsigned int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}
void print_float_matrix(float **mat, unsigned int m, unsigned int n) {

    unsigned int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            printf("%5.2f ", mat[i][j]);
        }
        printf("\n");
    }

}

unsigned int len = array_size(sbox);
unsigned int maxi = array_max(sbox, len);
unsigned int m = log_2(len);
unsigned int n = nbits(maxi);


// ==============================================================================

unsigned int **sbox_differential_table() {

    unsigned int **ddt;
    unsigned int nrows = two_power(m);
    unsigned int ncols = two_power(n);
    unsigned int i, si, di;

    ddt = alloc_uint_matrix(nrows, ncols);

    for (i = 0; i < nrows; ++i) {
        si = sbox[i];
        for (di = 0; di < nrows; ++di) {
            ddt[di][si^sbox[i^di]] += 1;
        }
    }

    return ddt;

}

unsigned int **sbox_linear_approx_table() {

    // the LAT is in original values (not subtracted by 2^n / 2)

    unsigned int **lat;
    unsigned int nrows = two_power(m);
    unsigned int ncols = two_power(n);
    unsigned int input, inputMask, outputMask;

    lat = alloc_uint_matrix(nrows, ncols);

    for (input = 0; input < nrows; ++input) {
        for (inputMask = 0; inputMask < nrows; ++inputMask) {
            for (outputMask = 0; outputMask < ncols; ++outputMask) {
                // x1 ^ x2 ^ x3 ...  = parity of x1x2x3...
                if (parity(input & inputMask) == parity(sbox[input] & outputMask)) {
                    lat[inputMask][outputMask] += 1;
                }
            }
        }
    }

    return lat;

}

int sbox_linearity() {

    // https://www.cosic.esat.kuleuven.be/ecrypt/courses/mykonos12/slides/day2/gl.pdf#59

    unsigned int **lat = sbox_linear_approx_table();
    unsigned int nrows = two_power(m);
    unsigned int ncols = two_power(n);
    unsigned int i, j;
    int val, maxAbs = 0;

    // skip zero input/output mask
    for (i = 1; i < nrows; ++i) {
        for (j = 1; j < ncols; ++j) {
            val = lat[i][j] - (two_power(m) / 2);
            val=abs(val);
            if (maxAbs < val)
                maxAbs = val;
        }
    }

    free(lat);

    return maxAbs;

}

int sbox_nonlinearity() {

    // the easy way
    // - https://github.com/okazymyrov/sbox/blob/master/Sage/CSbox.sage#L842
    // - http://crypto.stackexchange.com/a/19976

    int lin = sbox_linearity();
    return two_power(m - 1) - lin;

}


float *sbox_ac() {
    // Assuming m and sbox are defined appropriately
    int m = 16; // Example value
    unsigned int two_power_m = 1 << m; // Calculate 2^m

    float *k_aval = new float[m](); // Initialize with zero

    for (int i = 0; i < m; i++) {
        unsigned int ei = 1 << i; // Calculate 2^i
        for (unsigned int X = 0; X < two_power_m; ++X) {
            unsigned int dei = sbox[X] ^ sbox[X ^ ei];
            int w = __builtin_popcount(dei); // Calculate Hamming weight
            k_aval[i] += w;
        }
    }

    float div = (float)two_power_m * m;
    for (int i = 0; i < m; ++i) {
        k_aval[i] /= div;
    }

    return k_aval;
}



float **sbox_sac_matrix() {

    float **sac = alloc_float_matrix(m, n);
    unsigned int i, j, X, ei, ej, dei;

    for (i = 0; i < m; ++i) {
        ei = two_power(i);
        for (j = 0; j < n; ++j) {
            ej = two_power(j);
            for (X = 0; X < two_power(m); ++X) {
                dei = sbox[X] ^ sbox[X ^ ei];
                sac[i][j] += (dei & ej) >> j; // increment sac[i][j] if bit at position j of dei is set
            }
        }
    }

    float outputLength = (float)two_power(n);
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            sac[i][j] /= outputLength;
        }
    }

    return sac;

}

double sbox_bic() {

    unsigned int i, ei, X, dei;
    unsigned int j, k, ej, ek, dej, dek;
    unsigned int *aval_vector_j;
    unsigned int *aval_vector_k;
    double corr = 0.0;
    double maxCorr = 0.0;

    // for each input bit position
    for (i = 0; i < m; ++i) {
        ei = two_power(i);

        // for each j, k output bit change if j != k
        for (j = 0; j < n; ++j) {
            for (k = 0; k < n; ++k) {
                if (j != k) {
                    aval_vector_j = new unsigned int[two_power(m)]();
                    aval_vector_k = new unsigned int[two_power(m)]();

                    // for each possible input
                    for (X = 0; X < two_power(m); ++X) {
                        ej = two_power(j);
                        ek = two_power(k);

                        dei = sbox[X] ^ sbox[X ^ ei];
                        dej = (dei & ej) >> j;
                        dek = (dei & ek) >> k;

                        aval_vector_j[X] = dej;
                        aval_vector_k[X] = dek;
                    }

                    corr = fabs(correlation(aval_vector_j, aval_vector_k, two_power(m)));
                    // std::cout << "corr[" << j << "][" << k << "] = " << corr << std::endl;
                    if (maxCorr < corr)
                        maxCorr = corr;

                    delete[] aval_vector_j;
                    delete[] aval_vector_k;
                }
            }
        }
    }

    return maxCorr;
}


int main()
{
    Chaotic_SBox();
    Chaotic_Inv_S_Box();

    // Define an array of byte values
    byte plain[16] = {0x32, 0x88, 0x31, 0xe0,
                      0x43, 0x5a, 0x31, 0x37,
                      0xf6, 0x30, 0x98, 0x07,
                      0xa8, 0x8d, 0xa2, 0x34};

    byte referenceOutput[16];
    memcpy(referenceOutput, plain, sizeof(plain));
    aes(referenceOutput);

    /*
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
    */

    unsigned int **ddt = sbox_differential_table();

    std::cout << "DDT:" << std::endl;
    print_uint_matrix(ddt, two_power(m), two_power(n));

    unsigned int **lat = sbox_linear_approx_table();

    std::cout << "LAT:" << std::endl;
    print_uint_matrix(lat, two_power(m), two_power(n));

    float *ac = sbox_ac();

    std::cout << "AC:" << std::endl;
    for (int i = 0; i < m; ++i) {
        std::cout << " " << ac[i];
    }
    std::cout << std::endl;

    float **sac = sbox_sac_matrix();

    std::cout << "SAC:" << std::endl;
    print_float_matrix(sac, m, n);

    // Free the allocated memory
    free_uint_matrix(lat, two_power(m));
    free_uint_matrix(ddt, two_power(m));
    free_float_matrix(sac, m);
    free(ac);

    double bic = sbox_bic();
    std::cout << "BIC = " << bic << std::endl;

    int linearity = sbox_linearity();
    std::cout << "Linearity = " << linearity << std::endl;

    int nonlinearity = sbox_nonlinearity();
    std::cout << "Non-Linearity = " << nonlinearity << std::endl;


    return 0;
}
