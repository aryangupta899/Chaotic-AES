#include <iostream>
#include <bitset>
#include <random>
#include <vector>
using namespace std;

bitset<32> HashFunction(const std::bitset<32>& input)
{
    return input ^ 0b11001100;
}

int CountSetBits(const std::bitset<32>& bitset) {
    return bitset.count();
}

// Function to calculate the avalanche coefficient (K)
double CalculateAvalancheCoefficient(int numBits, int outputSize, int M)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> bitDistribution(0, numBits - 1);

    double totalAvalancheCoefficient = 0.0;

    for (int i = 0; i < M; ++i)
    {
        // Step 1: Generate a random number A1 with the size of input bits
        bitset<32> A1;
        for (int j = 0; j < numBits; ++j) {
            A1[j] = rd() % 2;
        }

        //Step 2: Calculate H1 = F(A1)
        bitset<32> H1 = HashFunction(A1);       //Yahan Hash Function Ki jagah AES use krna

        // Step 3: Toggle a random bit in A1 to generate A2
        int toggleIndex = bitDistribution(gen);
        bitset<32> A2 = A1;
        A2[toggleIndex] = !A2[toggleIndex];

        // Step 4: Calculate H2 = F(A2)
        bitset<32> H2 = HashFunction(A2);

        // Step 5: Compute X = H1 XOR H2
        std::bitset<32> X = H1 ^ H2;

        // Step 6: Calculate the number of set bits N in X
        int N = CountSetBits(X);

        // Step 7: Calculate the avalanche coefficient K = N / n
        double K = static_cast<double>(N) / static_cast<double>(outputSize);

        // Step 8: Accumulate K for later averaging
        totalAvalancheCoefficient += K;
    }

    // Calculate the mean avalanche coefficient
    double meanAvalancheCoefficient = totalAvalancheCoefficient / static_cast<double>(M);

    return meanAvalancheCoefficient;
}

int main() {
    const int numBits = 32; // Number of input bits
    const int outputSize = 32; // Number of output bits (size of hash output)
    const int M = 250000; // Number of iterations

    double meanAvalancheCoefficient = CalculateAvalancheCoefficient(numBits, outputSize, M);

    std::cout << "Mean Avalanche Coefficient (K): " << meanAvalancheCoefficient << std::endl;

    return 0;
}
