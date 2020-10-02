

# Rationality implies generalization
This is code accompanying the ICLR 2021 submission "For self-supervised learning, Rationality implies Generalization, provably"

We prove non-vacuous generalization bounds for SSS learning algorithms i.e. algorithms that learn by   
(i) performing pre-training with a self-supervised task (i.e., without labels) to obtain a complex representation of the data points, and then   
(ii) fitting a simple (e.g., linear) classifier on the representation and the labels.

In this repository, we assume that the self-supervised pre-trained model is available, and the representations are available. This repository shows how to compute the RRM bound for any given representation.

## Training the simple classifier
For each SSS-algorithm, calculating the RRM bound requires running two experiments  
1) the clean experiment where we train the simple classifier on the data and labels $(x, y)$  
2) the $\eta$-noisy experiment where we train the simple classifier on $(x, \tilde{y})$ where $\tilde{y}$ are the $\eta$ noised labels. 

#### Clean run


```python
%run fitlabels.py --train_noise_prob 0.0 --dataname CIFAR10 --feature_path ./data --log_predictions --batch_size 512 --epochs 100 --eval_type linear --from_features ----weight_decay 1e-06 --optimname adam --lr_sched_type const --lr 0.0002 --beta1 0.8 --beta2 0.999	
```

#### Noisy run


```python
%run fitlabels.py --train_noise_prob 0.05  --dataname CIFAR10 --feature_path ./data --log_predictions --batch_size 512 --epochs 100 --eval_type linear --from_features ----weight_decay 1e-06 --optimname adam --lr_sched_type const --lr 0.0002 --beta1 0.8 --beta2 0.999	
```

We compute these simple classifiers for a variety of self-supervised training methods. Given the training and test accuracies of both these runs, we compute the empirical RRM bound. 

# Computing Theorem II bound
We provide a theoretical bound for the Memorization gap in Theorem II. This bound can be computed empirically as follows. 

1. Compute K noisy runs using the code above  
2. Create the $N \times K$ matrix of the classifier predictions, where $N$ is the number of samples and $K$ is the number of trials called `pred_matrix`
3. Create the $N \times K$ matrices of the clean labels and noisy labels respectivly called `y_matrix` and `y_tilde_matrix` respectively.

### Complexity measure $C^{dc}$

```python
from complexity_functions import complexity, complexity_average
```


```python
num_classes = 10
noise_matrix = (y_tilde_matrix - y_matrix) % num_classes
diff_matrix = (pred_matrix - y_matrix) % num_classes 

Cdc = complexity_average(diff_matrix, noise_matrix)
Cdc = np.maximum(mi_acc_j_only, 0)

bound_Cdc = (np.mean(np.sqrt(0.5*Cdc)))/ 0.05

print(f'Bound based on Cdc is {bound_Cdc}')    
```

### Complexity measure $C^pc$


```python
Cpc = complexity(diff_matrix, noise_matrix)
Cpc = np.maximum(mi_acc_j_only, 0)

bound_Cpc = (np.sqrt(0.5*np.mean(Cpc)))/ 0.05

print(f'Bound based on Cpc is {bound_Cpc}') 
```

# RRM bound for CIFAR-10
We now list the various quantities of interest for CIFAR-10


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Backbone</th>
      <th>Data Augmentation</th>
      <th>Generalization Gap</th>
      <th>Robustness</th>
      <th>Memorization</th>
      <th>Rationality</th>
      <th>Theorem II bound</th>
      <th>RRM bound</th>
      <th>Test Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>amdim</td>
      <td>amdim_encoder</td>
      <td>False</td>
      <td>6.682000</td>
      <td>2.076349</td>
      <td>5.688700</td>
      <td>0.000000</td>
      <td>70.516720</td>
      <td>7.765049</td>
      <td>87.380000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>amdim</td>
      <td>resnet101</td>
      <td>False</td>
      <td>12.458000</td>
      <td>1.220833</td>
      <td>14.264408</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>15.485241</td>
      <td>62.430000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>amdim</td>
      <td>resnet18</td>
      <td>False</td>
      <td>4.338000</td>
      <td>0.422667</td>
      <td>4.581044</td>
      <td>0.000000</td>
      <td>33.470433</td>
      <td>5.003710</td>
      <td>62.280000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>amdim</td>
      <td>resnet50_bn</td>
      <td>False</td>
      <td>14.731333</td>
      <td>1.809750</td>
      <td>16.625074</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>18.434824</td>
      <td>66.283333</td>
    </tr>
    <tr>
      <th>20</th>
      <td>amdim</td>
      <td>wide_resnet50_2</td>
      <td>False</td>
      <td>13.070667</td>
      <td>1.698750</td>
      <td>15.327215</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>17.025965</td>
      <td>63.803333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mocov2</td>
      <td>resnet101</td>
      <td>False</td>
      <td>2.821333</td>
      <td>0.329500</td>
      <td>3.032190</td>
      <td>0.000000</td>
      <td>22.779988</td>
      <td>3.361690</td>
      <td>69.080000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mocov2</td>
      <td>resnet18</td>
      <td>False</td>
      <td>1.425333</td>
      <td>0.150250</td>
      <td>1.243309</td>
      <td>0.031775</td>
      <td>14.144346</td>
      <td>1.425333</td>
      <td>67.596667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>mocov2</td>
      <td>resnet50</td>
      <td>False</td>
      <td>2.718667</td>
      <td>0.296083</td>
      <td>2.964104</td>
      <td>0.000000</td>
      <td>24.181311</td>
      <td>3.260187</td>
      <td>70.086667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mocov2</td>
      <td>wide_resnet50_2</td>
      <td>False</td>
      <td>3.106667</td>
      <td>0.384917</td>
      <td>2.791697</td>
      <td>0.000000</td>
      <td>22.386794</td>
      <td>3.176614</td>
      <td>70.843333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>simclr</td>
      <td>resnet18</td>
      <td>False</td>
      <td>1.434000</td>
      <td>0.283048</td>
      <td>0.791300</td>
      <td>0.359652</td>
      <td>13.349844</td>
      <td>1.434000</td>
      <td>82.496667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>simclr</td>
      <td>resnet50</td>
      <td>False</td>
      <td>1.974000</td>
      <td>0.215833</td>
      <td>0.784471</td>
      <td>0.973696</td>
      <td>15.745243</td>
      <td>1.974000</td>
      <td>92.003333</td>
    </tr>
    <tr>
      <th>11</th>
      <td>simclr</td>
      <td>resnet50</td>
      <td>False</td>
      <td>2.240000</td>
      <td>0.520000</td>
      <td>1.711757</td>
      <td>0.008243</td>
      <td>19.532210</td>
      <td>2.240000</td>
      <td>84.943333</td>
    </tr>
    <tr>
      <th>17</th>
      <td>amdim</td>
      <td>amdim_encoder</td>
      <td>True</td>
      <td>4.430000</td>
      <td>0.682200</td>
      <td>0.356427</td>
      <td>3.391373</td>
      <td>10.323196</td>
      <td>4.430000</td>
      <td>87.326667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>amdim</td>
      <td>resnet101</td>
      <td>True</td>
      <td>-0.908600</td>
      <td>0.642133</td>
      <td>3.698682</td>
      <td>0.000000</td>
      <td>25.993151</td>
      <td>4.340815</td>
      <td>63.563333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>amdim</td>
      <td>resnet18</td>
      <td>True</td>
      <td>0.331400</td>
      <td>0.229575</td>
      <td>1.148386</td>
      <td>0.000000</td>
      <td>8.660545</td>
      <td>1.377961</td>
      <td>62.843333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>amdim</td>
      <td>resnet50_bn</td>
      <td>True</td>
      <td>3.693067</td>
      <td>0.837233</td>
      <td>4.222282</td>
      <td>0.000000</td>
      <td>31.119562</td>
      <td>5.059515</td>
      <td>66.440000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>amdim</td>
      <td>wide_resnet50_2</td>
      <td>True</td>
      <td>1.600533</td>
      <td>0.685423</td>
      <td>2.462525</td>
      <td>0.000000</td>
      <td>19.200017</td>
      <td>3.147948</td>
      <td>64.383333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mocov2</td>
      <td>resnet101</td>
      <td>True</td>
      <td>-6.013333</td>
      <td>0.152892</td>
      <td>0.706704</td>
      <td>0.000000</td>
      <td>6.377163</td>
      <td>0.859596</td>
      <td>68.576667</td>
    </tr>
    <tr>
      <th>0</th>
      <td>mocov2</td>
      <td>resnet18</td>
      <td>True</td>
      <td>-7.350733</td>
      <td>0.068200</td>
      <td>0.214771</td>
      <td>0.000000</td>
      <td>3.469925</td>
      <td>0.282971</td>
      <td>67.190000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mocov2</td>
      <td>resnet50</td>
      <td>True</td>
      <td>-5.381000</td>
      <td>0.189875</td>
      <td>0.836944</td>
      <td>0.000000</td>
      <td>6.986381</td>
      <td>1.026819</td>
      <td>69.683333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mocov2</td>
      <td>wide_resnet50_2</td>
      <td>True</td>
      <td>-6.371867</td>
      <td>0.180308</td>
      <td>1.026729</td>
      <td>0.000000</td>
      <td>7.632505</td>
      <td>1.207037</td>
      <td>70.993333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>simclr</td>
      <td>resnet50</td>
      <td>True</td>
      <td>-2.886267</td>
      <td>0.304940</td>
      <td>0.545692</td>
      <td>0.000000</td>
      <td>6.634170</td>
      <td>0.850632</td>
      <td>91.956667</td>
    </tr>
  </tbody>
</table>
</div>



# RRM bound for ImageNet



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Backbone</th>
      <th>Data Augmentation</th>
      <th>Generalization Gap</th>
      <th>Robustness</th>
      <th>Memorization</th>
      <th>Rationality</th>
      <th>Theorem II bound</th>
      <th>RRM bound</th>
      <th>Test Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>CMC</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>14.730569</td>
      <td>2.298659</td>
      <td>12.304347</td>
      <td>0.127563</td>
      <td>NaN</td>
      <td>14.730569</td>
      <td>54.596667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>InfoMin</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>10.207255</td>
      <td>2.343046</td>
      <td>8.963331</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>11.306377</td>
      <td>70.312667</td>
    </tr>
    <tr>
      <th>18</th>
      <td>InsDis</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>12.022083</td>
      <td>1.395160</td>
      <td>8.524625</td>
      <td>2.102298</td>
      <td>NaN</td>
      <td>12.022083</td>
      <td>56.673333</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PiRL</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>11.433350</td>
      <td>1.493768</td>
      <td>8.260058</td>
      <td>1.679524</td>
      <td>NaN</td>
      <td>11.433350</td>
      <td>59.105333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>amdim</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>13.624736</td>
      <td>0.902634</td>
      <td>9.715600</td>
      <td>3.006502</td>
      <td>NaN</td>
      <td>13.624736</td>
      <td>67.693000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bigbigan</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>29.595812</td>
      <td>3.132483</td>
      <td>25.189973</td>
      <td>1.273357</td>
      <td>NaN</td>
      <td>29.595812</td>
      <td>50.238667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>moco</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>10.718562</td>
      <td>1.822505</td>
      <td>7.860507</td>
      <td>1.035550</td>
      <td>NaN</td>
      <td>10.718562</td>
      <td>68.390667</td>
    </tr>
    <tr>
      <th>15</th>
      <td>simclr</td>
      <td>ResNet50_1x</td>
      <td>False</td>
      <td>11.071524</td>
      <td>1.218472</td>
      <td>7.727698</td>
      <td>2.125353</td>
      <td>NaN</td>
      <td>11.071524</td>
      <td>68.725333</td>
    </tr>
    <tr>
      <th>16</th>
      <td>simclrv2</td>
      <td>ResNet-50</td>
      <td>False</td>
      <td>11.164953</td>
      <td>0.639183</td>
      <td>7.674531</td>
      <td>2.851239</td>
      <td>NaN</td>
      <td>11.164953</td>
      <td>74.987333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>simclrv2</td>
      <td>r101_1x_sk0</td>
      <td>False</td>
      <td>10.528165</td>
      <td>1.113542</td>
      <td>6.992656</td>
      <td>2.421967</td>
      <td>NaN</td>
      <td>10.528165</td>
      <td>73.044000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>simclrv2</td>
      <td>r101_1x_sk1</td>
      <td>False</td>
      <td>8.234167</td>
      <td>0.709457</td>
      <td>4.663610</td>
      <td>2.861099</td>
      <td>NaN</td>
      <td>8.234167</td>
      <td>76.067333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>simclrv2</td>
      <td>r101_2x_sk0</td>
      <td>False</td>
      <td>11.024481</td>
      <td>0.736880</td>
      <td>7.512353</td>
      <td>2.775247</td>
      <td>NaN</td>
      <td>11.024481</td>
      <td>76.720000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>simclrv2</td>
      <td>r152_1x_sk0</td>
      <td>False</td>
      <td>10.316767</td>
      <td>1.120541</td>
      <td>6.932093</td>
      <td>2.264134</td>
      <td>NaN</td>
      <td>10.316767</td>
      <td>74.171333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>simclrv2</td>
      <td>r152_2x_sk0</td>
      <td>False</td>
      <td>10.924102</td>
      <td>0.753688</td>
      <td>7.445563</td>
      <td>2.724851</td>
      <td>NaN</td>
      <td>10.924102</td>
      <td>77.247333</td>
    </tr>
    <tr>
      <th>11</th>
      <td>simclrv2</td>
      <td>r50_1x_sk0</td>
      <td>False</td>
      <td>10.621827</td>
      <td>0.993703</td>
      <td>7.314382</td>
      <td>2.313741</td>
      <td>NaN</td>
      <td>10.621827</td>
      <td>70.693333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>InfoMin</td>
      <td>ResNet-50</td>
      <td>True</td>
      <td>4.882868</td>
      <td>0.807126</td>
      <td>1.012511</td>
      <td>3.063231</td>
      <td>NaN</td>
      <td>4.882868</td>
      <td>72.286400</td>
    </tr>
    <tr>
      <th>6</th>
      <td>InsDis</td>
      <td>ResNet-50</td>
      <td>True</td>
      <td>6.845196</td>
      <td>0.254156</td>
      <td>1.128115</td>
      <td>5.462925</td>
      <td>NaN</td>
      <td>6.845196</td>
      <td>58.300800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PiRL</td>
      <td>ResNet-50</td>
      <td>True</td>
      <td>6.225963</td>
      <td>0.291859</td>
      <td>0.987236</td>
      <td>4.946868</td>
      <td>NaN</td>
      <td>6.225963</td>
      <td>60.559600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>moco</td>
      <td>ResNet-50</td>
      <td>True</td>
      <td>1.316271</td>
      <td>0.565968</td>
      <td>0.927215</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.493183</td>
      <td>70.153600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>simclrv2</td>
      <td>r101_2x_sk0</td>
      <td>True</td>
      <td>0.633055</td>
      <td>0.103505</td>
      <td>0.804871</td>
      <td>0.000000</td>
      <td>47.90</td>
      <td>0.908376</td>
      <td>77.243600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>simclrv2</td>
      <td>r152_2x_sk0</td>
      <td>True</td>
      <td>1.004486</td>
      <td>0.130411</td>
      <td>0.772675</td>
      <td>0.101400</td>
      <td>NaN</td>
      <td>1.004486</td>
      <td>77.649600</td>
    </tr>
    <tr>
      <th>0</th>
      <td>simclrv2</td>
      <td>r50_1x_sk0</td>
      <td>True</td>
      <td>-2.336561</td>
      <td>0.261075</td>
      <td>0.675008</td>
      <td>0.000000</td>
      <td>46.93</td>
      <td>0.936083</td>
      <td>70.962400</td>
    </tr>
  </tbody>
</table>
</div>


