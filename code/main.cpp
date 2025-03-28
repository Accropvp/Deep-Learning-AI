#include <iostream>
#include <iomanip>
#include <sstream> // Include this for std::ostringstream
#include <cmath>
#include <windows.h>

using namespace std;
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
class Matrix
{
public:
    int rows, cols;
    double **data;

    // Default constructor
    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(int num_row, int num_col)
    {
        rows = num_row;
        cols = num_col;
        data = new double *[rows];
        for (int i = 0; i < rows; i++)
        {
            data[i] = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                data[i][j] = 0;
            }
        }
    }

    // Copy constructor
    Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
    {
        data = new double *[rows];
        for (int i = 0; i < rows; i++)
        {
            data[i] = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                data[i][j] = other.data[i][j];
            }
        }
    }

    Matrix &operator=(const Matrix &other)
    {
        if (this != &other) // Check for self-assignment
        {
            // Clean up existing data
            if (data != nullptr)
            {
                for (int i = 0; i < rows; i++)
                {
                    delete[] data[i];
                }
                delete[] data;
                data = nullptr; // Set to nullptr to avoid double deletion
            }
            rows = other.rows;
            cols = other.cols;
            data = new double *[rows];
            for (int i = 0; i < rows; i++)
            {
                data[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    data[i][j] = other.data[i][j];
                }
            }
        }
        return *this; // Return a reference to this object
    }

    Matrix operator+=(const Matrix &other)
    {
        if (data == nullptr)
        {
            rows = other.rows;
            cols = other.cols;
            data = new double *[rows];
            for (int i = 0; i < rows; i++)
            {
                data[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    data[i][j] = other.data[i][j];
                }
            }
            return *this; // Return a reference to this object
        }
        // Check if dimensions match
        if (rows != other.rows || cols != other.cols)
        {
            throw invalid_argument("Matrices must have the same dimensions for addition.");
        }
        if (rows == other.rows && cols == other.cols)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[i][j] += other.data[i][j];
                }
            }
        }
        return *this;
    }

    Matrix operator+(const Matrix &other)
    {
        // Check if dimensions match
        if (rows != other.rows || cols != other.cols)
        {
            throw invalid_argument("Matrices must have the same dimensions for addition.");
        }
        // Create a new Matrix to hold the result
        Matrix result(rows, cols);
        // Perform element-wise addition
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this->data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    Matrix operator*(const Matrix &other)
    {
        // Check if dimensions match
        if (cols != other.rows)
        {
            throw invalid_argument("Matrices must have compatible dimensions for multiplication.");
        }
        // Create a new Matrix to hold the result
        Matrix result(rows, other.cols);
        // Perform matrix multiplication
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < other.cols; j++)
            {
                for (int k = 0; k < cols; k++)
                {
                    result.data[i][j] += this->data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(const double &other)
    {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this->data[i][j] * other;
            }
        }
        return result;
    }

    Matrix operator*(const int &other)
    {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this->data[i][j] * other;
            }
        }
        return result;
    }

    bool operator!=(const Matrix &other) const
    {
        // First, check if dimensions match
        if (rows != other.rows || cols != other.cols)
        {
            return true; // Matrices are not equal if dimensions are different
        }

        // Check each element for inequality
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (data[i][j] != other.data[i][j])
                {
                    return true; // Matrices are not equal if any element differs
                }
            }
        }
        return false; // Matrices are equal
    }

    Matrix ReLU()
    {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = max(0.0, data[i][j]);
            }
        }
        return result;
    }

    Matrix normalize()
    {
        Matrix result(rows, cols);
        double sum = 0.0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sum += data[i][j];
            }
        }
        for (int k = 0; k < rows; k++)
        {
            for (int l = 0; l < cols; l++)
            {
                result.data[k][l] = data[k][l] / sum;
            }
        }
        return result;
    }

    double maximal()
    {
        double max = data[0][0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (data[i][j] > max)
                    max = data[i][j];
            }
        }
        return max;
    }

    ~Matrix()
    {
        // cout << "destructor in process" << endl;
        if (data != nullptr)
        {
            for (int i = 0; i < rows; i++)
            {
                delete[] data[i];
            }
            delete[] data;
            data = nullptr;
        }
        // cout << "destructor process compleate" << endl;
    }

    // Existing constructor, operator overloads, and destructor...

    // Method to convert the matrix to a string
    string toString() const
    {
        ostringstream oss; // Create an output string stream
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                oss << data[i][j]; // Add the current element to the stream
                if (j < cols - 1)
                {
                    oss << ", "; // Add a comma if not the last element in the row
                }
            }
            if (i < rows - 1)
            {
                oss << "\n"; // Add a newline after each row
            }
        }
        return oss.str(); // Return the constructed string
    }
};

class NeuralNetwork
{
    int numberLayers;
    Matrix inputLayer;
    Matrix *weights;
    Matrix *biases;

public:
    // Default constructor
    NeuralNetwork() : inputLayer(Matrix(1, 1)), weights(nullptr), biases(nullptr) {}

    NeuralNetwork(int numLayers)
    {
        numberLayers = numLayers;        // Store the number of layers
        weights = new Matrix[numLayers]; // Allocate memory for the weights matrices
        biases = new Matrix[numLayers];  // Allocate memory for the bias matrices
    }

    NeuralNetwork(int numLayers, Matrix *WeightLayers, Matrix *BiasLayers)
    {
        numberLayers = numLayers;        // Store the number of layers
        weights = new Matrix[numLayers]; // Allocate memory for the weights matrices
        biases = new Matrix[numLayers];  // Allocate memory for the bias matrices
        for (int i = 0; i < numLayers; i++)
        {
            weights[i] = WeightLayers[i]; // Initialize the weights matrices
            biases[i] = BiasLayers[i];    // Initialize the bias matrices
        }
    }

    NeuralNetwork(int numLayers, Matrix InpLayer, Matrix *WeightLayers, Matrix *BiasLayers)
    {
        numberLayers = numLayers;        // Store the number of layers
        weights = new Matrix[numLayers]; // Allocate memory for the weights matrices
        biases = new Matrix[numLayers];  // Allocate memory for the bias matrices
        inputLayer = InpLayer;           // Copy the input layer
        for (int i = 0; i < numLayers; i++)
        {
            weights[i] = WeightLayers[i]; // Copy the weights matrices
            biases[i] = BiasLayers[i];    // Copy the bias matrices
        }
    }
    int getNumberOfLayers()
    {
        return numberLayers;
    }
    Matrix *getWeight()
    {
        return weights;
    }
    Matrix *getBias()
    {
        return biases;
    }

    void defNumOfLayer(int num)
    {
        numberLayers = num;
        cout << "Warning : changing the number of layersvariable is not recomended. \n"
             << "Please make sure you know what you are doing" << endl;
    }

    void setInputLayer(Matrix inpLayer)
    {
        inputLayer = inpLayer;
    }

    void defWeight(Matrix *inpWeights)
    {
        for (int i = 0; i < numberLayers - 1; i++)
        {
            weights[i] = inpWeights[i];
        }
    }

    void defBias(Matrix *inpBias)
    {
        for (int i = 0; i < numberLayers - 1; i++)
        {
            biases[i] = inpBias[i];
        }
    }

    Matrix forwardPropagation(Matrix neuronLayer, Matrix layerWeight, Matrix layerBias)
    {
        Matrix result = (layerWeight * neuronLayer) + layerBias;
        return result.ReLU();
    }

    Matrix *fullForwardPropagation()
    {
        Matrix *neuron = new Matrix[numberLayers];
        Matrix currentLayer = inputLayer;
        for (int i = 0; i < numberLayers; i++)
        {
            currentLayer = forwardPropagation(currentLayer, weights[i], biases[i]);
            neuron[i] = currentLayer;
        }
        return neuron;
    }

    Matrix outputLayer(Matrix *neuron)
    {
        return neuron[numberLayers - 1];
    }

    Matrix neroneCost(Matrix output, Matrix expectedValues)
    {
        if (output.rows != expectedValues.rows)
        {
            throw invalid_argument("the output layer must have the same number of rows than the expected values");
        }
        Matrix result(output.rows, 1);
        for (int i = 0; i < output.rows; i++)
        {
            result.data[i][0] = pow(output.data[i][0] - expectedValues.data[i][0], 2);
        }
        return result;
    }

    int cost(Matrix output, Matrix expectedValue) // cost function of the network
    {
        if (output.rows != expectedValue.rows)
        {
            throw invalid_argument("the output layer must have the same number of rows than the expected values");
        }
        double result = 0;
        for (int i = 0; i < output.rows; i++)
        {
            result += pow(output.data[i][0] - expectedValue.data[i][0], 2);
        }
        return result;
    }

    Matrix derivedCost(Matrix output, Matrix expectedValues)
    {
        Matrix result(output.rows, output.cols);
        for (int i = 0; i < output.rows; i++)
        {
            result.data[i][0] = 2 * (output.data[i][0] - expectedValues.data[i][0]);
        }
        return result;
    }

    Matrix *derivedNerones(Matrix *InitialNeuroneData, Matrix dCost)
    {
        Matrix *result = new Matrix[numberLayers];
        result[numberLayers - 1] = dCost;
        // iterate for every layer backward
        for (int i = numberLayers - 1; i > 0; i--)
        {
            result[i - 1] = Matrix(InitialNeuroneData[i - 1].rows, 1);
            for (int k = 0; k < InitialNeuroneData[i - 1].rows; k++)
            {
                double sum = 0.0;
                // derivative of Cost / retro Neurone
                for (int j = 0; j < result[i].rows; j++)
                {
                    // Check if the activation is positive (ReLU derivative)
                    if (InitialNeuroneData[i].data[j][0] > 0)
                    {
                        // Derivative of cost/nerone
                        double derivCost = result[i].data[j][0];
                        // Derivative of nerone/old neurone
                        double derivWeight = weights[i].data[j][k];
                        sum += derivCost * derivWeight;
                    }
                }
                result[i - 1].data[k][0] = sum;
            }
        }
        return result;
    }

    Matrix *derivedNerones(Matrix *InitialNeuroneData, Matrix *InitialWeight, Matrix dCost, int numOfLayers)
    {
        Matrix *result = new Matrix[numOfLayers];
        result[numOfLayers - 1] = dCost;
        // iterate for every layer backward
        for (int i = numOfLayers - 1; i > 0; i--)
        {
            result[i - 1] = Matrix(InitialNeuroneData[i - 1].rows, 1);
            for (int k = 0; k < InitialNeuroneData[i - 1].rows; k++)
            {
                double sum = 0.0;
                // derivative of Cost / retro Neurone
                for (int j = 0; j < result[i].rows; j++)
                {
                    // Check if the activation is positive (ReLU derivative)
                    if (InitialNeuroneData[i].data[j][0] > 0)
                    {
                        // Derivative of cost/nerone
                        double derivCost = result[i].data[j][0];
                        // Derivative of nerone/old neurone
                        double derivWeight = InitialWeight[i].data[j][k];
                        sum += derivCost * derivWeight;
                    }
                }
                result[i - 1].data[k][0] = sum;
            }
        }
        return result;
    }

    Matrix *derivedWeights(Matrix *derived_neurone_data, Matrix *neuron)
    {
        Matrix *result = new Matrix[numberLayers];

        // derivative of the first layer of weights
        result[0] = Matrix(derived_neurone_data[0].rows, inputLayer.rows);
        for (int k = 0; k < inputLayer.rows; k++)
        {
            for (int j = 0; j < derived_neurone_data[0].rows; j++)
            {
                // Check if the activation is positive (ReLU derivative)
                if (neuron[0].data[j][0] > 0)
                {
                    result[0].data[j][k] = derived_neurone_data[0].data[j][0] * inputLayer.data[k][0];
                }
                else
                {
                    result[0].data[j][k] = 0;
                }
            }
        }

        for (int i = 1; i < numberLayers; i++)
        {
            // derivative of a layer of weights
            result[i] = Matrix(derived_neurone_data[i].rows, neuron[i - 1].rows);
            for (int k = 0; k < neuron[i - 1].rows; k++)
            {
                // derivative of weight
                for (int j = 0; j < derived_neurone_data[i].rows; j++)
                {
                    // Check if the activation is positive (ReLU derivative)
                    if (neuron[i].data[j][0] > 0)
                    {
                        result[i].data[j][k] = derived_neurone_data[i].data[j][0] * neuron[i - 1].data[k][0];
                    }
                    else
                    {
                        result[i].data[j][k] = 0;
                    }
                }
            }
        }
        return result;
    }

    Matrix *derivedWeights(Matrix *derived_neurone_data, Matrix *Initial_Neurone_Data, Matrix inpLayer, int numOfLayers)
    {
        Matrix *result = new Matrix[numOfLayers];

        // derivative of the first layer of weights
        result[0] = Matrix(derived_neurone_data[0].rows, inpLayer.rows);
        for (int k = 0; k < inpLayer.rows; k++)
        {
            for (int j = 0; j < derived_neurone_data[0].rows; j++)
            {
                // Check if the activation is positive (ReLU derivative)
                if (Initial_Neurone_Data[0].data[j][0] > 0)
                {
                    result[0].data[j][k] = derived_neurone_data[0].data[j][0] * inpLayer.data[k][0];
                }
                else
                {
                    result[0].data[j][k] = 0;
                }
            }
        }

        for (int i = 1; i < numOfLayers; i++)
        {
            // derivative of a layer of weights
            result[i] = Matrix(derived_neurone_data[i].rows, Initial_Neurone_Data[i - 1].rows);
            for (int k = 0; k < Initial_Neurone_Data[i - 1].rows; k++)
            {
                // derivative of weight
                for (int j = 0; j < derived_neurone_data[i].rows; j++)
                {
                    // Check if the activation is positive (ReLU derivative)
                    if (Initial_Neurone_Data[i].data[j][0] > 0)
                    {
                        result[i].data[j][k] = derived_neurone_data[i].data[j][0] * Initial_Neurone_Data[i - 1].data[k][0];
                    }
                    else
                    {
                        result[i].data[j][k] = 0;
                    }
                }
            }
        }
        return result;
    }

    Matrix *derivedBiases(Matrix *derived_data, Matrix *neuron)
    {
        Matrix *result = new Matrix[numberLayers];
        for (int i = 0; i < numberLayers; i++)
        {
            // derivative of a layer of bias
            result[i] = Matrix(derived_data[i].rows, 1);

            for (int j = 0; j < derived_data[i].rows; j++)
            {
                // Check if the activation is positive (ReLU derivative)
                if (neuron[i].data[j][0] > 0)
                {
                    result[i].data[j][0] = derived_data[i].data[j][0];
                }
                else
                {
                    result[i].data[j][0] = 0;
                }
            }
        }
        return result;
    }

    Matrix *derivedBiases(Matrix *derived_data, Matrix *Initial_Neuron_Data, int numOfLayers)
    {
        Matrix *result = new Matrix[numOfLayers];
        for (int i = 0; i < numOfLayers; i++)
        {
            // derivative of a layer of bias
            result[i] = Matrix(derived_data[i].rows, 1);

            for (int j = 0; j < derived_data[i].rows; j++)
            {
                // Check if the activation is positive (ReLU derivative)
                if (Initial_Neuron_Data[i].data[j][0] > 0)
                {
                    result[i].data[j][0] = derived_data[i].data[j][0];
                }
                else
                {
                    result[i].data[j][0] = 0;
                }
            }
        }
        return result;
    }

    Matrix **avgCostGradient(Matrix *input, Matrix *expectedOutput, int numOfIteration)
    {
        // give the avreage weight cost gradient then the avreage bias cost gradient
        Matrix **result = new Matrix *[2];
        Matrix *gradientWeights = new Matrix[numberLayers];
        Matrix *gradientBiases = new Matrix[numberLayers];
        // Calculate the average cost
        for (int i = 0; i < numOfIteration; i++)
        {
            setInputLayer(input[i]);                                                       // set the input layer
            Matrix *neuron = fullForwardPropagation();                                     // Process with forward propagation
            Matrix dCost = derivedCost(neuron[numberLayers - 1], expectedOutput[i]);       // calculate the derivative of the cost of each ouput
            Matrix *dNerones = derivedNerones(neuron, weights, dCost, numberLayers);       // derivative of every nerones
            Matrix *dWeights = derivedWeights(dNerones, neuron, inputLayer, numberLayers); // derivative of every weights
            Matrix *dBiases = derivedBiases(dNerones, neuron, numberLayers);               // derivative of every biases

            // Add the gradients to the result
            for (int j = 0; j < numberLayers; j++)
            {
                gradientWeights[j] += dWeights[j];
                gradientBiases[j] += dBiases[j];
            }
        }
        float invNumOfIteration = 1.0 / numOfIteration;
        for (int i = 0; i < numberLayers; i++)
        {
            gradientWeights[i] = gradientWeights[i] * invNumOfIteration;
            gradientBiases[i] = gradientBiases[i] * invNumOfIteration;
        }

        result[0] = gradientWeights;
        result[1] = gradientBiases;

        return result;
    }

    Matrix **avgCostGradient(Matrix *input, Matrix *expectedOutput, int numOfLayer, int numOfIteration)
    {
        // give the avreage weight cost gradient then the avreage bias cost gradient
        Matrix **result = new Matrix *[2];
        Matrix *gradientWeights = new Matrix[numOfLayer];
        Matrix *gradientBiases = new Matrix[numOfLayer];
        // Calculate the average cost
        for (int i = 0; i < numOfIteration; i++)
        {
            setInputLayer(input[i]);                                                     // set the input layer
            Matrix *neuron = fullForwardPropagation();                                   // Process with forward propagation
            Matrix dCost = derivedCost(neuron[numOfLayer - 1], expectedOutput[i]);       // calculate the derivative of the cost of each ouput
            Matrix *dNerones = derivedNerones(neuron, weights, dCost, numOfLayer);       // derivative of every nerones
            Matrix *dWeights = derivedWeights(dNerones, neuron, inputLayer, numOfLayer); // derivative of every weights
            Matrix *dBiases = derivedBiases(dNerones, neuron, numOfLayer);               // derivative of every biases

            // Add the gradients to the result
            for (int j = 0; j < numOfLayer; j++)
            {
                gradientWeights[j] += dWeights[j];
                gradientBiases[j] += dBiases[j];
            }
        }
        float invNumOfIteration = 1.0 / numOfIteration;
        for (int i = 0; i < numOfLayer; i++)
        {
            gradientWeights[i] = gradientWeights[i] * invNumOfIteration;
            gradientBiases[i] = gradientBiases[i] * invNumOfIteration;
        }

        result[0] = gradientWeights;
        result[1] = gradientBiases;

        return result;
    }

    void decreaseCostFunc(Matrix **costGradient, int precision)
    {
        Matrix *weightGradient = costGradient[0];
        Matrix *biasGradient = costGradient[1];
        // get the greatest gradient number
        // get a tiny nudge to the negative gradient of the cost function
        Matrix *negativeGradientWeights = new Matrix[numberLayers];
        Matrix *negativeGradientBiases = new Matrix[numberLayers];
        int nega_div = (-1 / precision);
        for (int i = 0; i < numberLayers - 1; i++)
        {
            negativeGradientWeights[i] = weightGradient[i] * nega_div;
            negativeGradientBiases[i] = biasGradient[i] * nega_div;

            // add the tiny nudge to the weight and biases to get the steepest decrease of the cost function
            weights[i] = weights[i] + negativeGradientWeights[i];
            biases[i] = biases[i] + negativeGradientBiases[i];
        }
    }
};

class Test
{
public:
    void failedPrint()
    {
        cout << '(';
        // set the colorattribute to red
        SetConsoleTextAttribute(hConsole, 12);
        cout << "Failed";
        // set the colorattribute to white
        SetConsoleTextAttribute(hConsole, 15);
        cout << ')' << endl;
    }

    void passedPrint()
    {
        cout << '(';
        // set the colorattribute to red
        SetConsoleTextAttribute(hConsole, 10);
        cout << "Passed";
        // set the colorattribute to white
        SetConsoleTextAttribute(hConsole, 15);
        cout << ')' << endl;
    }

    void testPrint(bool test)
    {
        if (test)
            passedPrint();
        else
            failedPrint();
    }

    bool testMatrixInequality()
    {
        Matrix m1(3, 2);
        Matrix m2(2, 3);
        Matrix m3(3, 2);

        for (int i = 0; i < 3; i++)
        {
            m1.data[i][0] = i + 1;
            m2.data[0][i] = 2 * (i + 1);
            m3.data[i][0] = 3 * (i + 1);

            m1.data[i][1] = 3 * (i + 1);
            m2.data[1][i] = 4 * (i + 1);
            m3.data[i][1] = 5 * (i + 1);
        }

        // test inequality
        bool testInequality = false;
        if (m1 != m2 || m1 != m3)
            testInequality = true;
        if (m1 != m1 || m2 != m2)
            testInequality = false;

        cout << "Matrix inequality test : ";
        testPrint(testInequality);
        return testInequality;
    }

    bool testMatrixAssigment()
    {
        Matrix m1(3, 2);
        for (int i = 0; i < 3; i++)
        {
            m1.data[i][0] = i + 1;
            m1.data[i][1] = 3 * (i + 1);
        }
        // test assignement
        bool testAssignement = true;
        Matrix m2 = m1;
        if (m2 != m1)
            testAssignement = false;
        cout << "Matrix assignement test : ";
        testPrint(testAssignement);
        return testAssignement;
    }

    bool testMatrixAdditionAssignment()
    {
        Matrix m1(3, 2);
        for (int i = 0; i < 3; i++)
        {
            m1.data[i][0] = i + 1;
            m1.data[i][1] = 3 * (i + 1);
        }
        Matrix m2(3, 2);
        for (int i = 0; i < 3; i++)
        {
            m2.data[i][0] = 2 * (i + 1);
            m2.data[i][1] = 4 * (i + 1);
        }
        // test assignement
        bool testAssignement = true;
        Matrix m0;
        m0 += m1;
        m0 += m2;
        for (int i = 0; i < 3; i++)
        {
            if (m0.data[i][0] = !(i + 1 + 2 * (i + 1)))
                testAssignement = false;

            if (m0.data[i][1] = !(3 * (i + 1) + 4 * (i + 1)))
                testAssignement = false;
        }

        cout << "Matrix addition assignement test : ";
        testPrint(testAssignement);
        return testAssignement;
    }

    bool testMatrixAddition()
    {
        Matrix m1(3, 2);
        Matrix m2(3, 2);

        for (int i = 0; i < 3; i++)
        {
            m1.data[i][0] = i + 1;
            m2.data[i][0] = 3 * (i + 1);

            m1.data[i][1] = 3 * (i + 1);
            m2.data[i][1] = 5 * (i + 1);
        }
        // test addition
        bool testAddition = true;
        Matrix m3 = m1 + m2;

        for (int i = 0; i < 3; i++)
        {
            if (m3.data[i][0] != 4 * (i + 1) || m3.data[i][1] != 8 * (i + 1))
                testAddition = false;
        }
        cout << "Matrix additon test : ";
        testPrint(testAddition);
        return testAddition;
    }

    bool testMatrixMultiplication()
    {
        Matrix m1(3, 2);
        Matrix m2(2, 3);

        for (int i = 0; i < 3; i++)
        {
            m1.data[i][0] = i + 1;
            m2.data[0][i] = 2 * (i + 1);

            m1.data[i][1] = 3 * (i + 1);
            m2.data[1][i] = 4 * (i + 1);
        }
        // test multiplication
        bool testMulti = true;
        Matrix m3 = m1 * m2;
        Matrix m4(3, 3);
        if (m3.rows != m1.rows || m3.cols != m2.cols)
        {
            testMulti = false;
        }
        m4.data[0][0] = 14;
        m4.data[0][1] = 28;
        m4.data[0][2] = 42;
        m4.data[1][0] = 28;
        m4.data[1][1] = 56;
        m4.data[1][2] = 84;
        m4.data[2][0] = 42;
        m4.data[2][1] = 84;
        m4.data[2][2] = 126;

        if (m3 != m4)
            testMulti = false;
        cout << "Matrix multiplication test : ";
        testPrint(testMulti);

        // test Matrix multiplication to a constant
        bool testMultiConst = true;
        Matrix m5 = m1 * 2;
        Matrix m6(3, 2);
        m6.data[0][0] = 2;
        m6.data[1][0] = 4;
        m6.data[2][0] = 6;
        m6.data[0][1] = 6;
        m6.data[1][1] = 12;
        m6.data[2][1] = 18;
        if (m5 != m6)
            testMultiConst = false;
        cout << "Matrix multiplication to int test : ";
        testPrint(testMultiConst);

        // test Matrix multiplication to a decimal constant
        bool testMultiDeciConst = true;
        Matrix m7 = m1 * 2.5;
        Matrix m8(3, 2);
        m8.data[0][0] = 2.5;
        m8.data[1][0] = 5;
        m8.data[2][0] = 7.5;
        m8.data[0][1] = 7.5;
        m8.data[1][1] = 15;
        m8.data[2][1] = 22.5;
        if (m7 != m8)
            testMultiDeciConst = false;
        cout << "Matrix multiplication to double test : ";
        testPrint(testMultiDeciConst);
        return testMulti && testMultiConst && testMultiDeciConst;
    }

    bool testMatrixReLU()
    {
        // test the ReLU function
        bool testReLU = true;
        Matrix m1(3, 3);
        m1.data[0][0] = -1;
        m1.data[0][1] = 1;
        m1.data[0][2] = 0;
        m1.data[1][0] = -16;
        m1.data[1][1] = 16;
        m1.data[1][2] = 0;
        m1.data[2][0] = -256;
        m1.data[2][1] = 256;
        m1.data[2][2] = 0;
        Matrix m2 = m1.ReLU();
        Matrix m3(3, 3);
        m3.data[0][0] = 0;
        m3.data[0][1] = 1;
        m3.data[0][2] = 0;
        m3.data[1][0] = 0;
        m3.data[1][1] = 16;
        m3.data[1][2] = 0;
        m3.data[2][0] = 0;
        m3.data[2][1] = 256;
        m3.data[2][2] = 0;
        if (m2 != m3)
            testReLU = false;

        cout << "Matrix ReLU test : ";
        testPrint(testReLU);

        return testReLU;
    }

    bool testMatrixNormalize()
    {
        // test normalize function
        bool testNormalize = false;
        Matrix m12(3, 2);
        m12.data[0][0] = 1;
        m12.data[0][1] = 2;
        m12.data[1][0] = 3;
        m12.data[1][1] = 4;
        m12.data[2][0] = 5;
        m12.data[2][1] = 6;
        Matrix m13 = m12.normalize();
        double sum = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                sum += m13.data[i][j];
            }
        }

        if (0.99 < sum && sum < 1.01)
            testNormalize = true;
        cout << "Matrix normalize test : ";
        testPrint(testNormalize);
        return testNormalize;
    }

    bool testMatrixMaximal()
    {
        // test the maximal function
        bool testMax = false;
        Matrix m1(3, 3);
        m1.data[0][0] = 1;
        m1.data[0][1] = 2;
        m1.data[0][2] = 3;
        m1.data[1][0] = 1;
        m1.data[1][1] = 17;
        m1.data[1][2] = 317;
        m1.data[2][0] = 1;
        m1.data[2][1] = 2;
        m1.data[2][2] = 317;
        double maxi = m1.maximal();
        if (maxi == 317)
            testMax = true;
        cout << "Matrix maximal test : ";
        testPrint(testMax);
        return testMax;
    }

    void testMatrix()
    {
        testMatrixInequality();
        testMatrixAssigment();
        testMatrixAdditionAssignment();
        testMatrixAddition();
        testMatrixMultiplication();
        testMatrixReLU();
        testMatrixNormalize();
        testMatrixMaximal();
    }

    bool testNetworkForwardProp()
    {

        NeuralNetwork steve(2);
        // test the forward propagation function
        bool testForward = false;
        Matrix input(4, 1);
        Matrix weight(4, 4);
        Matrix bias(4, 1);
        Matrix output(4, 1);

        input.data[0][0] = 1;
        input.data[1][0] = 0.7;
        input.data[2][0] = 0.2;
        input.data[3][0] = 0;

        weight.data[0][0] = 2;
        weight.data[1][0] = 4;
        weight.data[2][0] = 2.9;
        weight.data[3][0] = 8.2;
        weight.data[0][1] = 7.6;
        weight.data[1][1] = 8.4;
        weight.data[2][1] = 7.6;
        weight.data[3][1] = -1.1;
        weight.data[0][2] = 0;
        weight.data[1][2] = -5.5;
        weight.data[2][2] = -7.6;
        weight.data[3][2] = 3;
        weight.data[0][3] = -2;
        weight.data[1][3] = 4.1;
        weight.data[2][3] = 0.5;
        weight.data[3][3] = 7;

        bias.data[0][0] = 5.2;
        bias.data[1][0] = -9;
        bias.data[2][0] = 5.6;
        bias.data[3][0] = 7.2;

        output.data[0][0] = 12.52;
        output.data[1][0] = 0;
        output.data[2][0] = 12.3;
        output.data[3][0] = 15.23;

        Matrix fProp = steve.forwardPropagation(input, weight, bias);

        for (int i = 0; i < 4; i++)
        {
            double diff = output.data[i][0] - fProp.data[i][0];
            if (0.01 > diff && diff > -0.01)
            {
                testForward = true;
            }
        }

        cout << "Neral Network Forward Propagation test : ";
        testPrint(testForward);
        return testForward;
    }

    bool testNetworkFullForwardProp()
    {
        // test the forward propagation function
        bool testForwardFull = true;
        Matrix input(4, 1);
        Matrix weight(4, 4);
        Matrix bias(4, 1);
        Matrix output(4, 1);

        input.data[0][0] = 1;
        input.data[1][0] = 0.7;
        input.data[2][0] = 0.2;
        input.data[3][0] = 0;

        weight.data[0][0] = 2;
        weight.data[1][0] = 4;
        weight.data[2][0] = 2.9;
        weight.data[3][0] = 8.2;
        weight.data[0][1] = 7.6;
        weight.data[1][1] = 8.4;
        weight.data[2][1] = 7.6;
        weight.data[3][1] = -1.1;
        weight.data[0][2] = 0;
        weight.data[1][2] = -5.5;
        weight.data[2][2] = -7.6;
        weight.data[3][2] = 3;
        weight.data[0][3] = -2;
        weight.data[1][3] = 4.1;
        weight.data[2][3] = 0.5;
        weight.data[3][3] = 7;

        bias.data[0][0] = 5.2;
        bias.data[1][0] = -9;
        bias.data[2][0] = 5.6;
        bias.data[3][0] = 7.2;

        output.data[0][0] = 0;
        output.data[1][0] = 1331.1666;
        output.data[2][0] = 404.9218;
        output.data[3][0] = 1741.3577;

        Matrix *weights = new Matrix[3];
        Matrix *biases = new Matrix[3];

        for (int i = 0; i < 3; i++)
        {
            weights[i] = weight;
            biases[i] = bias;
        }
        NeuralNetwork steve(3, input, weights, biases);

        Matrix *neuron = steve.fullForwardPropagation();
        for (int i = 0; i < 4; i++)
        {
            double diff = output.data[i][0] - neuron[2].data[i][0];
            if (0.01 < diff || diff < -0.01)
            {
                testForwardFull = false;
            }
        }
        cout << "Neral Network Full Forward Propagation test : ";
        testPrint(testForwardFull);
        return testForwardFull;
    }

    bool testNetworkNeroneCost()
    {
        NeuralNetwork steve(2);
        Matrix out(5, 1);
        Matrix expect(5, 1);
        Matrix cost(5, 1);

        out.data[0][0] = 0.27;
        out.data[1][0] = 0.76;
        out.data[2][0] = 0.95;
        out.data[3][0] = 0.15;
        out.data[4][0] = 0.56;

        expect.data[0][0] = 0;
        expect.data[1][0] = 1;
        expect.data[2][0] = 1;
        expect.data[3][0] = 0;
        expect.data[4][0] = 1;

        cost.data[0][0] = 0.0729;
        cost.data[1][0] = 0.0576;
        cost.data[2][0] = 0.0025;
        cost.data[3][0] = 0.0225;
        cost.data[4][0] = 0.1936;

        Matrix result = steve.neroneCost(out, expect);
        bool testNetworkNeroneCost = true;

        for (int i = 0; i < 5; i++)
        {
            if (0.0001 < result.data[i][0] - cost.data[i][0] || result.data[i][0] - cost.data[i][0] < -0.0001)
            {
                testNetworkNeroneCost = false;
            }
        }
        cout << "Neral Network Neron Cost test : ";
        testPrint(testNetworkNeroneCost);
        return testNetworkNeroneCost;
    }

    bool testNetworkCost()
    {
        NeuralNetwork steve(2);
        Matrix out(5, 1);
        Matrix expect(5, 1);
        double cost = 0.3491;
        bool testNetworkCost = false;

        out.data[0][0] = 0.27;
        out.data[1][0] = 0.76;
        out.data[2][0] = 0.95;
        out.data[3][0] = 0.15;
        out.data[4][0] = 0.56;

        expect.data[0][0] = 0;
        expect.data[1][0] = 1;
        expect.data[2][0] = 1;
        expect.data[3][0] = 0;
        expect.data[4][0] = 1;

        double result = steve.cost(out, expect);

        if (0.0001 > result - cost || result - cost > -0.0001)
        {
            testNetworkCost = true;
        }

        cout << "Neral Network Cost test : ";
        testPrint(testNetworkCost);
        return testNetworkCost;
    }

    bool testNetworkDerivedCost()
    {
        NeuralNetwork steve(2);
        Matrix out(3, 1);
        Matrix expect(3, 1);
        Matrix derivedCost(3, 1);

        out.data[0][0] = 0.5;
        out.data[1][0] = 0.7;
        out.data[2][0] = 0.9;

        expect.data[0][0] = 1.0;
        expect.data[1][0] = 0.8;
        expect.data[2][0] = 0.6;

        derivedCost.data[0][0] = -1.0;
        derivedCost.data[1][0] = -0.2;
        derivedCost.data[2][0] = 0.6;

        Matrix result = steve.derivedCost(out, expect);
        bool testDerivedCost = true;

        for (int i = 0; i < 3; i++)
        {
            if (abs(result.data[i][0] - derivedCost.data[i][0]) > 0.0001)
            {
                testDerivedCost = false;
            }
        }
        cout << "Neral Network Derived Cost test : ";
        testPrint(testDerivedCost);
        return testDerivedCost;
    }

    bool testNetworkDerivedNeurones()
    {
        // Define input layer (3 neurons)
        Matrix input(3, 1);
        input.data[0][0] = 1.0;
        input.data[1][0] = 0.5;
        input.data[2][0] = 0.2;

        // Define weights for the first layer (3x3 matrix)
        Matrix weight1(3, 3);
        weight1.data[0][0] = 0.1;
        weight1.data[0][1] = 0.2;
        weight1.data[0][2] = 0.3;
        weight1.data[1][0] = 0.4;
        weight1.data[1][1] = 0.5;
        weight1.data[1][2] = 0.6;
        weight1.data[2][0] = 0.7;
        weight1.data[2][1] = 0.8;
        weight1.data[2][2] = 0.9;

        // Define weights for the second layer (3x3 matrix)
        Matrix weight2(3, 3);
        weight2.data[0][0] = 0.9;
        weight2.data[0][1] = 0.8;
        weight2.data[0][2] = 0.7;
        weight2.data[1][0] = 0.6;
        weight2.data[1][1] = 0.5;
        weight2.data[1][2] = 0.4;
        weight2.data[2][0] = 0.3;
        weight2.data[2][1] = 0.2;
        weight2.data[2][2] = 0.1;

        // Define biases for the first layer (3x1 matrix)
        Matrix bias1(3, 1);
        bias1.data[0][0] = 0.1;
        bias1.data[1][0] = 0.2;
        bias1.data[2][0] = 0.3;

        // Define biases for the second layer (3x1 matrix)
        Matrix bias2(3, 1);
        bias2.data[0][0] = 0.3;
        bias2.data[1][0] = 0.2;
        bias2.data[2][0] = 0.1;

        // Set up the neural network
        Matrix weights[2] = {weight1, weight2};
        Matrix biases[2] = {bias1, bias2};
        NeuralNetwork steve(2, input, weights, biases);

        // Perform forward propagation
        Matrix *neuron = steve.fullForwardPropagation();

        // Define expected output (3x1 matrix)
        Matrix expectedOutput(3, 1);
        expectedOutput.data[0][0] = 1.0;
        expectedOutput.data[1][0] = 0.0;
        expectedOutput.data[2][0] = 0.0;

        // Compute the derivative of the cost function with respect to the output layer
        Matrix dCost = steve.derivedCost(neuron[1], expectedOutput);
        Matrix *steve_weights = steve.getWeight();

        // Compute the derivatives of the neurons in each layer
        Matrix *dNeurones = steve.derivedNerones(neuron, steve_weights, dCost, 2);

        // Expected derivatives for each layer
        Matrix expectedDerivedLayer1(3, 1);
        expectedDerivedLayer1.data[0][0] = 4.8864;
        expectedDerivedLayer1.data[1][0] = 4.1666;
        expectedDerivedLayer1.data[2][0] = 3.4468;

        Matrix expectedDerivedLayer2(3, 1);
        expectedDerivedLayer2.data[0][0] = 3.012;
        expectedDerivedLayer2.data[1][0] = 3.066;
        expectedDerivedLayer2.data[2][0] = 1.12;

        // Check if the computed derivatives match the expected values
        bool testDerivedNeurones = true;

        // cout << "dNeurones 1 : " << dNeurones[0].toString() << endl;
        //  cout << "dNeurones 2 : " << dNeurones[1].toString() << endl;

        for (int i = 0; i < 3; i++)
        {
            if (abs(dNeurones[0].data[i][0] - expectedDerivedLayer1.data[i][0]) > 0.0001)
            {
                testDerivedNeurones = false;
            }
            if (abs(dNeurones[1].data[i][0] - expectedDerivedLayer2.data[i][0]) > 0.0001)
            {
                testDerivedNeurones = false;
            }
        }

        // Output the results of the test
        cout << "Neral Network Derived Neurones test : ";
        testPrint(testDerivedNeurones);

        // Clean up dynamically allocated memory
        delete[] dNeurones;

        return testDerivedNeurones;
    }

    bool testNetworkDerivedWeights()
    {
        // Define input layer (3 neurons)
        Matrix input(3, 1);
        input.data[0][0] = -10.4;
        input.data[1][0] = 6.0;
        input.data[2][0] = 0.2;

        // Define weights for the first layer (3x3 matrix)
        Matrix weight1(3, 3);
        weight1.data[0][0] = 0.1;
        weight1.data[0][1] = 0.2;
        weight1.data[0][2] = 0.3;
        weight1.data[1][0] = 0.4;
        weight1.data[1][1] = 0.5;
        weight1.data[1][2] = 0.6;
        weight1.data[2][0] = 0.7;
        weight1.data[2][1] = 0.8;
        weight1.data[2][2] = 0.9;

        // Define weights for the second layer (3x3 matrix)
        Matrix weight2(3, 3);
        weight2.data[0][0] = 0.9;
        weight2.data[0][1] = 0.8;
        weight2.data[0][2] = 0.7;
        weight2.data[1][0] = 0.6;
        weight2.data[1][1] = 0.5;
        weight2.data[1][2] = 0.4;
        weight2.data[2][0] = 0.3;
        weight2.data[2][1] = 0.2;
        weight2.data[2][2] = 0.1;

        // Define biases for the first layer (3x1 matrix)
        Matrix bias1(3, 1);
        bias1.data[0][0] = 0.1;
        bias1.data[1][0] = 0.2;
        bias1.data[2][0] = 0.3;

        // Define biases for the second layer (3x1 matrix)
        Matrix bias2(3, 1);
        bias2.data[0][0] = 0.3;
        bias2.data[1][0] = 0.2;
        bias2.data[2][0] = 0.1;

        // Set up the neural network
        Matrix weights[2] = {weight1, weight2};
        Matrix biases[2] = {bias1, bias2};
        NeuralNetwork steve(2, input, weights, biases);

        // Perform forward propagation
        Matrix *neuron = steve.fullForwardPropagation();
        Matrix output = steve.outputLayer(neuron);
        // Define expected output (3x1 matrix)
        Matrix expectedOutput(3, 1);
        expectedOutput.data[0][0] = 4.4;
        expectedOutput.data[1][0] = 2.1;
        expectedOutput.data[2][0] = 5.2;

        // Compute the derivative of the cost function with respect to the output layer
        Matrix dCost = steve.derivedCost(output, expectedOutput);

        Matrix *steve_weights = steve.getWeight();

        // Compute the derivatives of the neurons in each layer
        Matrix *dNeurones = steve.derivedNerones(neuron, steve_weights, dCost, 2);

        Matrix *dWeight = steve.derivedWeights(dNeurones, neuron, input, 2);

        Matrix expectedWeight1(3, 3);
        expectedWeight1.data[0][0] = 123.90144, expectedWeight1.data[0][1] = -71.4816, expectedWeight1.data[0][2] = -2.38272;
        expectedWeight1.data[1][0] = 0.0, expectedWeight1.data[1][1] = 0.0, expectedWeight1.data[1][2] = 0.0;
        expectedWeight1.data[2][0] = 0.0, expectedWeight1.data[2][1] = 0.0, expectedWeight1.data[2][2] = 0.0;

        Matrix expectedWeight2(3, 3);
        expectedWeight2.data[0][0] = -2.43968, expectedWeight2.data[0][1] = 0.0, expectedWeight2.data[0][2] = 0.0;
        expectedWeight2.data[1][0] = -1.09312, expectedWeight2.data[1][1] = 0.0, expectedWeight2.data[1][2] = 0.0;
        expectedWeight2.data[2][0] = -3.20256, expectedWeight2.data[2][1] = 0.0, expectedWeight2.data[2][2] = 0.0;

        bool testDW = true;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (abs(dWeight[0].data[i][j] - expectedWeight1.data[i][j]) > 0.0001)
                {
                    testDW = false;
                }
                if (abs(dWeight[1].data[i][j] - expectedWeight2.data[i][j]) > 0.0001)
                {
                    testDW = false;
                }
            }
        }

        // Output the results of the test
        cout << "Neral Network Derived Weights test : ";
        testPrint(testDW);

        return testDW;
    }

    bool testNetworkDerivedBiases()
    {
        // Define input layer (3 neurons)
        Matrix input(3, 1);
        input.data[0][0] = 1.0;
        input.data[1][0] = 0.5;
        input.data[2][0] = 0.2;

        // Define weights for the first layer (3x3 matrix)
        Matrix weight1(3, 3);
        weight1.data[0][0] = 0.1;
        weight1.data[0][1] = 0.2;
        weight1.data[0][2] = 0.3;
        weight1.data[1][0] = 0.4;
        weight1.data[1][1] = 0.5;
        weight1.data[1][2] = 0.6;
        weight1.data[2][0] = 0.7;
        weight1.data[2][1] = 0.8;
        weight1.data[2][2] = 0.9;

        // Define weights for the second layer (3x3 matrix)
        Matrix weight2(3, 3);
        weight2.data[0][0] = 0.9;
        weight2.data[0][1] = 0.8;
        weight2.data[0][2] = 0.7;
        weight2.data[1][0] = 0.6;
        weight2.data[1][1] = 0.5;
        weight2.data[1][2] = 0.4;
        weight2.data[2][0] = 0.3;
        weight2.data[2][1] = 0.2;
        weight2.data[2][2] = 0.1;

        // Define biases for the first layer (3x1 matrix)
        Matrix bias1(3, 1);
        bias1.data[0][0] = 0.1;
        bias1.data[1][0] = 0.2;
        bias1.data[2][0] = 0.3;

        // Define biases for the second layer (3x1 matrix)
        Matrix bias2(3, 1);
        bias2.data[0][0] = 0.3;
        bias2.data[1][0] = 0.2;
        bias2.data[2][0] = 0.1;

        // Set up the neural network
        Matrix weights[2] = {weight1, weight2};
        Matrix biases[2] = {bias1, bias2};
        NeuralNetwork steve(2, input, weights, biases);

        // Perform forward propagation
        Matrix *neuron = steve.fullForwardPropagation();
        Matrix output = steve.outputLayer(neuron);

        // Define expected output (3x1 matrix)
        Matrix expectedOutput(3, 1);
        expectedOutput.data[0][0] = 1.0;
        expectedOutput.data[1][0] = 0.0;
        expectedOutput.data[2][0] = 0.0;

        // Compute the derivative of the cost function with respect to the output layer
        Matrix dCost = steve.derivedCost(output, expectedOutput);

        Matrix *steve_weights = steve.getWeight();

        // Compute the derivatives of the neurons in each layer
        Matrix *dNeurones = steve.derivedNerones(neuron, steve_weights, dCost, 2);

        Matrix *dBiases = steve.derivedBiases(dNeurones, neuron, 2);

        Matrix exDB1(3, 1);
        exDB1.data[0][0] = 4.8864;
        exDB1.data[1][0] = 4.1666;
        exDB1.data[2][0] = 3.4468;
        Matrix exDB2(3, 1);
        exDB2.data[0][0] = 3.012;
        exDB2.data[1][0] = 3.066;
        exDB2.data[2][0] = 1.12;

        bool testDB = true;

        for (int i = 0; i < 3; i++)
        {
            if (abs(dBiases[0].data[i][0] - exDB1.data[i][0]) > 0.0001)
            {
                testDB = false;
            }
            if (abs(dBiases[1].data[i][0] - exDB2.data[i][0]) > 0.0001)
            {
                testDB = false;
            }
        }

        // Print the results of the test
        cout << "Neral Network Derived Biases test : ";
        testPrint(testDB);

        return testDB;
    }

    bool testAvgCostGradient()
    {
        // Define input layer (3 neurons)
        Matrix input(3, 1);
        input.data[0][0] = 1.0;
        input.data[1][0] = 0.5;
        input.data[2][0] = 0.2;

        // Define weights for the first layer (3x3 matrix)
        Matrix weight1(3, 3);
        weight1.data[0][0] = 0.1;
        weight1.data[0][1] = 0.2;
        weight1.data[0][2] = 0.3;
        weight1.data[1][0] = 0.4;
        weight1.data[1][1] = 0.5;
        weight1.data[1][2] = 0.6;
        weight1.data[2][0] = 0.7;
        weight1.data[2][1] = 0.8;
        weight1.data[2][2] = 0.9;

        // Define weights for the second layer (3x3 matrix)
        Matrix weight2(3, 3);
        weight2.data[0][0] = 0.9;
        weight2.data[0][1] = 0.8;
        weight2.data[0][2] = 0.7;
        weight2.data[1][0] = 0.6;
        weight2.data[1][1] = 0.5;
        weight2.data[1][2] = 0.4;
        weight2.data[2][0] = 0.3;
        weight2.data[2][1] = 0.2;
        weight2.data[2][2] = 0.1;

        // Define biases for the first layer (3x1 matrix)
        Matrix bias1(3, 1);
        bias1.data[0][0] = 0.1;
        bias1.data[1][0] = 0.2;
        bias1.data[2][0] = 0.3;

        // Define biases for the second layer (3x1 matrix)
        Matrix bias2(3, 1);
        bias2.data[0][0] = 0.3;
        bias2.data[1][0] = 0.2;
        bias2.data[2][0] = 0.1;

        // Set up the neural network
        Matrix weights[2] = {weight1, weight2};
        Matrix biases[2] = {bias1, bias2};
        NeuralNetwork steve(2, input, weights, biases);

        // Perform forward propagation
        Matrix *neuron = steve.fullForwardPropagation();

        Matrix input1(3, 1);
        input1.data[0][0] = 1.0;
        input1.data[1][0] = 2.0;
        input1.data[2][0] = 3.0;

        Matrix input2(3, 1);
        input2.data[0][0] = 4.0;
        input2.data[1][0] = 2.6;
        input2.data[2][0] = -1.5;

        Matrix input3(3, 1);
        input3.data[0][0] = -10.4;
        input3.data[1][0] = 6.0;
        input3.data[2][0] = 0.2;

        Matrix ExpectedOutput1(3, 1);
        ExpectedOutput1.data[0][0] = 1.0;
        ExpectedOutput1.data[1][0] = 0.0;
        ExpectedOutput1.data[2][0] = 1.0;

        Matrix ExpectedOutput2(3, 1);
        ExpectedOutput2.data[0][0] = 2.0;
        ExpectedOutput2.data[1][0] = 1.6;
        ExpectedOutput2.data[2][0] = 1.5;

        Matrix ExpectedOutput3(3, 1);
        ExpectedOutput3.data[0][0] = 4.4;
        ExpectedOutput3.data[1][0] = 2.1;
        ExpectedOutput3.data[2][0] = 5.2;

        Matrix inputs[] = {input1, input2, input3};
        Matrix outputs[] = {ExpectedOutput1, ExpectedOutput2, ExpectedOutput3};

        Matrix **gradient = steve.avgCostGradient(inputs, outputs, 3);

        Matrix *dWeights = gradient[0];
        Matrix *dBiases = gradient[1];

        Matrix Expected_dWeight1(3, 3);
        Expected_dWeight1.data[0][0] = 57.67168, Expected_dWeight1.data[0][1] = -4.58912, Expected_dWeight1.data[0][2] = 14.55856;
        Expected_dWeight1.data[1][0] = 14.3413333333333, Expected_dWeight1.data[1][1] = 16.7702666666667, Expected_dWeight1.data[1][2] = 13.243;
        Expected_dWeight1.data[2][0] = 12.3114666666667, Expected_dWeight1.data[2][1] = 14.3024533333333, Expected_dWeight1.data[2][2] = 11.1332;
        Matrix Expected_dWeight2(3, 3);
        Expected_dWeight2.data[0][0] = 7.50329333333333, Expected_dWeight2.data[0][1] = 20.8205333333333, Expected_dWeight2.data[0][2] = 33.3245466666667;
        Expected_dWeight2.data[1][0] = 5.15374666666667, Expected_dWeight2.data[1][1] = 13.4605333333333, Expected_dWeight2.data[1][2] = 21.4029466666667;
        Expected_dWeight2.data[2][0] = -0.461800000000001, Expected_dWeight2.data[2][1] = 1.1272, Expected_dWeight2.data[2][2] = 1.64868;
        Matrix Ex_dWeights[] = {Expected_dWeight1, Expected_dWeight2};

        Matrix Expected_dBias1(3, 1);
        Expected_dBias1.data[0][0] = 4.8976;
        Expected_dBias1.data[1][0] = 7.72333333333333;
        Expected_dBias1.data[2][0] = 6.57786666666667;
        Matrix Expected_dBias2(3, 1);
        Expected_dBias2.data[0][0] = 4.348;
        Expected_dBias2.data[1][0] = 3.19066666666667;
        Expected_dBias2.data[2][0] = -3.1;
        Matrix Ex_dBiases[] = {Expected_dBias1, Expected_dBias2};

        bool testACG = true;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (abs(Ex_dBiases[i].data[j][0] - dBiases[i].data[j][0]) > 0.0001)
                {
                    // cout << "dBias " << i << " : " << j << ", 0 : " << abs(Ex_dBiases[i].data[j][0] - dBiases[i].data[j][0]) << " :\n"
                    //      << dBiases[i].toString() << '\n';
                    testACG = false;
                    break;
                }
                for (int k = 0; k < 3; k++)
                {
                    if (abs(Ex_dWeights[i].data[j][k] - dWeights[i].data[j][k]) > 0.0001)
                    {
                        // cout << "dWeight " << i << " : " << j << ", " << k << " : " << abs(Ex_dWeights[i].data[j][k] - dWeights[i].data[j][k]) << ", " << Ex_dWeights[i].data[j][k] << ", " << dWeights[i].data[j][k] << " :\n"
                        //      << dWeights[i].toString() << '\n';
                        testACG = false;
                        break;
                    }
                }
            }
        }
        // Print the results of the test
        cout << "Neral Network Average cost gradient test : ";
        testPrint(testACG);
        return testACG;
    }

    void testNetwork()
    {
        testNetworkForwardProp();
        testNetworkFullForwardProp();
        testNetworkNeroneCost();
        testNetworkCost();
        testNetworkDerivedCost();
        testNetworkDerivedNeurones();
        testNetworkDerivedWeights();
        testNetworkDerivedBiases();
        testAvgCostGradient();
    }
};

int main()
{
    // Initialize a 3x3 matrix
    Matrix myMatrix(4, 2);

    // Optionally, you can set values in the matrix
    myMatrix.data[0][0] = 1.0;
    myMatrix.data[0][1] = 2.0;
    myMatrix.data[1][0] = 3.0;
    myMatrix.data[1][1] = 4.0;
    myMatrix.data[2][0] = 5.0;
    myMatrix.data[2][1] = 6.0;
    myMatrix.data[3][0] = 7.0;
    myMatrix.data[3][1] = 8.0;

    // Initialize a 3x3 matrix
    Matrix myOtherMatrix(2, 4);

    // Optionally, you can set values in the matrix
    myOtherMatrix.data[0][0] = 1.0;
    myOtherMatrix.data[0][1] = 2.0;
    myOtherMatrix.data[0][2] = 3.0;
    myOtherMatrix.data[0][3] = 4.0;
    myOtherMatrix.data[1][0] = 5.0;
    myOtherMatrix.data[1][1] = 6.0;
    myOtherMatrix.data[1][2] = 7.0;
    myOtherMatrix.data[1][3] = 8.0;

    Matrix myResultMatrix = myMatrix * myOtherMatrix;
    Test test;

    int num;

    cout << "test output\n";
    test.failedPrint();
    test.passedPrint();
    test.testMatrix();
    test.testNetwork();
    cout << "test output 2\n";
    system("pause");
    return 0;
}