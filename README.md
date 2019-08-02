# Breast-cancer-prediction-ML-Python
Make predictions for breast cancer, malignant or benign using the Breast Cancer data set<br>
Dataset - Breast Cancer Wisconsin (Original) Data Set<br>
This code demonstrates logistic regression on the dataset and also uses gradient descent to lower the BCE(binary cross entropy).
## <strong>Dataset description</strong>

![](/pictures/breast%20cancer%20description.PNG)
<ol>
<li>Sample code number: id number</li>
<li>Clump Thickness: 1 - 10</li>
<li>Uniformity of Cell Size: 1 - 10</li>
<li>Uniformity of Cell Shape: 1 - 10</li>
<li>Marginal Adhesion: 1 - 10</li>
<li>Single Epithelial Cell Size: 1 - 10</li>
<li>Bare Nuclei: 1 - 10</li>
<li>Bland Chromatin: 1 - 10</li>
<li>Normal Nucleoli: 1 - 10</li>
<li>Mitoses: 1 - 10</li>
<li>Class: (2 for benign, 4 for malignant)</li>
</ol>
<h2>Libraries required</h2>
<ol>
  <li>numpy
      <br>
        <code>pip install numpy</code>
    </li>
  <li>pandas
    <br>
    <code>
      pip install pandas
    </code>
  </li>
  <li>
    random
    <br>
    <code>
      pip install random
    </code>
  </li>
  <li>
    seaborn
    <br>
    <code>
      pip install seaborn
    </code>
  </li>
</ol>
<h2> Logistic regression algorithm </h2>

![](/pictures/logistic_regression.gif)

<ul>
<li>
  Use the sigmoid activation function - <img src="https://latex.codecogs.com/gif.latex?$\sigma(z)=&space;1/1&plus;e^{-z}$"            title="$\sigma(z)= 1/1+e^{-z}$" />
  </li>
 <li> 
  Remember the gradient descent formula for liner regression where Mean squared error was used but we cannot use Mean squared error here so replace with some error <img src="https://latex.codecogs.com/gif.latex?$E$" title="$E$" />
  </li>
  <li>
    Gradient Descent - <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{j}=\theta&space;_{j}-&space;\alpha\cdot&space;\partial&space;MSE\partial&space;/&space;\theta&space;_{j}&space;$" title="\theta _{j}=\theta _{j}- \alpha\cdot \partial MSE\partial / \theta _{j} $" />
    Logistic regression - <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{j}=\theta&space;_{j}-&space;\alpha\cdot&space;\partial&space;E\partial&space;/&space;\theta&space;_{j}&space;$" title="\theta _{j}=\theta _{j}- \alpha\cdot \partial E\partial / \theta _{j} $" />
  </li>
  <li>
    Conditions for E:
    <ol>
      <li> Convex or as convex as possible</li>
      <li> Should be function of <img src="https://latex.codecogs.com/gif.latex?$\theta$" title="$\theta$" /></li>
      <li> Should be differentiable</li>
      </ol>
  </li>
  <li>
    So use, Entropy = <img src="https://latex.codecogs.com/gif.latex?$-p&space;\log&space;p$" title="$-p \log p$" />
  </li>
  <li>As we cant use both <img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /> and y so use cross entropy
  <img src="https://latex.codecogs.com/gif.latex?-y&space;\log&space;\hat{y}" title="-y \log \hat{y}" /> as
    <img src="https://latex.codecogs.com/gif.latex?\hat{y}\epsilon&space;[0,1]" title="\hat{y}\epsilon [0,1]" />
  </li>
  <li>
    So add 2 cross entropies CE 1 = <img src="https://latex.codecogs.com/gif.latex?-y&space;\log&space;\hat{y}" title="-y \log \hat{y}" /> and CE 2 = <img src="https://latex.codecogs.com/gif.latex?(1-y)&space;\log&space;(1-&space;\hat{y})" title="(1-y) \log (1- \hat{y})" />.
    We get Binary Cross entropy (BCE) = <img src="https://latex.codecogs.com/gif.latex?-y\log&space;\hat{y}-(1-y)&space;\log&space;(1-&space;\hat{y})" title="-y\log \hat{y}-(1-y) \log (1- \hat{y})" />
    <li>
      So now our formula becomes,
      <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{j}=\theta&space;_{j}-&space;\alpha\cdot&space;\partial&space;BCE/&space;\partial&space;\theta&space;_{j}&space;$" title="\theta _{j}=\theta _{j}- \alpha\cdot \partial BCE/ \partial \theta _{j} $" />
      <li>
        Using simple chain rule we obtain,
        <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{j}=\theta&space;_{j}-&space;\frac{\alpha}{m}\cdot&space;(\hat&space;y&space;-&space;y)^{T}\cdot&space;X" title="\theta _{j}=\theta _{j}- \frac{\alpha}{m}\cdot (\hat y - y)^{T}\cdot X" />
        </li>
      </li>
    </li>
    <li>
  Now apply Gradient Descent with this formula
  </li>
</ul>
