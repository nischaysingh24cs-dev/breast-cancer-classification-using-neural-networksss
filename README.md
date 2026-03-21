{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Breast Cancer Classification with a simple Neural Network (NN)**"
      ],
      "metadata": {
        "id": "RNC9RB0Non0k"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Previous videos:\n",
        "\n",
        "\n",
        "*   Brest Cancer Classification with Logistic Regression video: https://youtu.be/bFh1umUDaGc\n",
        "*   EDA video: https://youtu.be/imdS1LIlISY\n",
        "*   Data Visualization video: https://youtu.be/gFSerq21wo8\n",
        "\n"
      ],
      "metadata": {
        "id": "7GUR8W1Iowy3"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "X3wT8l6lfj--"
      },
      "source": [
        "**Importing the Dependencies**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XqsQmOXGXXTe"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import sklearn.datasets\n",
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pwJ9zLukg3Q_"
      },
      "source": [
        "Data Collection & Processing"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j6bMZMKUgz7L"
      },
      "source": [
        "# loading the data from sklearn\n",
        "breast_cancer_dataset = sklearn.datasets.load_breast_cancer()"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xdY6i73KgkDG",
        "outputId": "2a879655-8f19-4f31-9e45-888b0a8cffba"
      },
      "source": [
        "print(breast_cancer_dataset)"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,\n",
            "        1.189e-01],\n",
            "       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,\n",
            "        8.902e-02],\n",
            "       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,\n",
            "        8.758e-02],\n",
            "       ...,\n",
            "       [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,\n",
            "        7.820e-02],\n",
            "       [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,\n",
            "        1.240e-01],\n",
            "       [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,\n",
            "        7.039e-02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,\n",
            "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,\n",
            "       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,\n",
            "       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
            "       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
            "       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
            "       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 'frame': None, 'target_names': array(['malignant', 'benign'], dtype='<U9'), 'DESCR': '.. _breast_cancer_dataset:\\n\\nBreast cancer wisconsin (diagnostic) dataset\\n--------------------------------------------\\n\\n**Data Set Characteristics:**\\n\\n:Number of Instances: 569\\n\\n:Number of Attributes: 30 numeric, predictive attributes and the class\\n\\n:Attribute Information:\\n    - radius (mean of distances from center to points on the perimeter)\\n    - texture (standard deviation of gray-scale values)\\n    - perimeter\\n    - area\\n    - smoothness (local variation in radius lengths)\\n    - compactness (perimeter^2 / area - 1.0)\\n    - concavity (severity of concave portions of the contour)\\n    - concave points (number of concave portions of the contour)\\n    - symmetry\\n    - fractal dimension (\"coastline approximation\" - 1)\\n\\n    The mean, standard error, and \"worst\" or largest (mean of the three\\n    worst/largest values) of these features were computed for each image,\\n    resulting in 30 features.  For instance, field 0 is Mean Radius, field\\n    10 is Radius SE, field 20 is Worst Radius.\\n\\n    - class:\\n            - WDBC-Malignant\\n            - WDBC-Benign\\n\\n:Summary Statistics:\\n\\n===================================== ====== ======\\n                                        Min    Max\\n===================================== ====== ======\\nradius (mean):                        6.981  28.11\\ntexture (mean):                       9.71   39.28\\nperimeter (mean):                     43.79  188.5\\narea (mean):                          143.5  2501.0\\nsmoothness (mean):                    0.053  0.163\\ncompactness (mean):                   0.019  0.345\\nconcavity (mean):                     0.0    0.427\\nconcave points (mean):                0.0    0.201\\nsymmetry (mean):                      0.106  0.304\\nfractal dimension (mean):             0.05   0.097\\nradius (standard error):              0.112  2.873\\ntexture (standard error):             0.36   4.885\\nperimeter (standard error):           0.757  21.98\\narea (standard error):                6.802  542.2\\nsmoothness (standard error):          0.002  0.031\\ncompactness (standard error):         0.002  0.135\\nconcavity (standard error):           0.0    0.396\\nconcave points (standard error):      0.0    0.053\\nsymmetry (standard error):            0.008  0.079\\nfractal dimension (standard error):   0.001  0.03\\nradius (worst):                       7.93   36.04\\ntexture (worst):                      12.02  49.54\\nperimeter (worst):                    50.41  251.2\\narea (worst):                         185.2  4254.0\\nsmoothness (worst):                   0.071  0.223\\ncompactness (worst):                  0.027  1.058\\nconcavity (worst):                    0.0    1.252\\nconcave points (worst):               0.0    0.291\\nsymmetry (worst):                     0.156  0.664\\nfractal dimension (worst):            0.055  0.208\\n===================================== ====== ======\\n\\n:Missing Attribute Values: None\\n\\n:Class Distribution: 212 - Malignant, 357 - Benign\\n\\n:Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\\n\\n:Donor: Nick Street\\n\\n:Date: November, 1995\\n\\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\\nhttps://goo.gl/U2Uwz2\\n\\nFeatures are computed from a digitized image of a fine needle\\naspirate (FNA) of a breast mass.  They describe\\ncharacteristics of the cell nuclei present in the image.\\n\\nSeparating plane described above was obtained using\\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, \"Decision Tree\\nConstruction Via Linear Programming.\" Proceedings of the 4th\\nMidwest Artificial Intelligence and Cognitive Science Society,\\npp. 97-101, 1992], a classification method which uses linear\\nprogramming to construct a decision tree.  Relevant features\\nwere selected using an exhaustive search in the space of 1-4\\nfeatures and 1-3 separating planes.\\n\\nThe actual linear program used to obtain the separating plane\\nin the 3-dimensional space is that described in:\\n[K. P. Bennett and O. L. Mangasarian: \"Robust Linear\\nProgramming Discrimination of Two Linearly Inseparable Sets\",\\nOptimization Methods and Software 1, 1992, 23-34].\\n\\nThis database is also available through the UW CS ftp server:\\n\\nftp ftp.cs.wisc.edu\\ncd math-prog/cpo-dataset/machine-learn/WDBC/\\n\\n.. dropdown:: References\\n\\n  - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extrac...more
