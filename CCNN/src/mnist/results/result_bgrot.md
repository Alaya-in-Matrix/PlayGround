# results from cross-validation + l1

## first try

* learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
* regularization_strengths = [1e5,1e4,1e3,1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

### learning_rate = 0.1

* reg = 100000.0 , error = 0.90032
* reg = 10000.0 , error = 0.90032
* reg = 1000.0 , error = 0.90218
* reg = 100.0 , error = 0.89548
* reg = 10.0 , error = 0.90218
* reg = 1 , error = 0.89862
* reg = 0.1 , error = 0.89794
* reg = 0.01 , error = 0.88878
* reg = 0.001 , error = 0.88878
* reg = 0.0001 , error = 0.67910004
* reg = 1e-05 , error = 0.90034
* reg = 1e-06 , error = 0.90032
* reg = 1e-07 , error = 0.90032

### learning_rate = 0.01

* reg = 100000.0 , error = 0.88878
* reg = 10000.0 , error = 0.88878
* reg = 1000.0 , error = 0.88878
* reg = 100.0 , error = 0.33532
* reg = 10.0 , error = 0.34748
* reg = 1 , error = 0.34892
* reg = 0.1 , error = 0.36052
* reg = 0.01 , error = 0.34924
* reg = 0.001 , error = 0.35722
* reg = 0.0001 , error = 0.34766
* reg = 1e-05 , error = 0.35884
* reg = 1e-06 , error = 0.36596
* reg = 1e-07 , error = 0.36089998

### learning_rate = 0.001

* reg = 100000.0 , error = 0.88878
* reg = 10000.0 , error = 0.88878
* reg = 1000.0 , error = 0.88878
* reg = 100.0 , error = 0.34175998
* reg = 10.0 , error = 0.32488
* reg = 1 , error = 0.32845998
* reg = 0.1 , error = 0.32525998
* reg = 0.01 , error = 0.32687998
* reg = 0.001 , error = 0.33186
* reg = 0.0001 , error = 0.3279
* reg = 1e-05 , error = 0.3207
* reg = 1e-06 , error = 0.33091998
* reg = 1e-07 , error = 0.3243

### learning_rate = 0.0001

* reg = 100000.0 , error = 0.88878
* reg = 10000.0 , error = 0.88878
* reg = 1000.0 , error = 0.88878
* reg = 100.0 , error = 0.49904
* reg = 10.0 , error = 0.47087997
* reg = 1 , error = 0.45884
* reg = 0.1 , error = 0.47575998
* reg = 0.01 , error = 0.4536
* reg = 0.001 , error = 0.50119996
* reg = 0.0001 , error = 0.45712
* reg = 1e-05 , error = 0.44304
* reg = 1e-06 , error = 0.47726
* reg = 1e-07 , error = 0.48180002

### learning_rate = 1e-05

* reg = 100000.0 , error = 0.88878
* reg = 10000.0 , error = 0.88878
* reg = 1000.0 , error = 0.88878
* reg = 100.0 , error = 0.75348
* reg = 10.0 , error = 0.75038
* reg = 1 , error = 0.75522
* reg = 0.1 , error = 0.7522
* reg = 0.01 , error = 0.74702
* reg = 0.001 , error = 0.74656
* reg = 0.0001 , error = 0.75181997
* reg = 1e-05 , error = 0.7408
* reg = 1e-06 , error = 0.75478
* reg = 1e-07 , error = 0.74994004

### learning_rate = 1e-06

* reg = 100000.0 , error = 0.8934
* reg = 10000.0 , error = 0.87428
* reg = 1000.0 , error = 0.88774
* reg = 100.0 , error = 0.87752
* reg = 10.0 , error = 0.89264
* reg = 1 , error = 0.91272
* reg = 0.1 , error = 0.87632
* reg = 0.01 , error = 0.87262
* reg = 0.001 , error = 0.88945997
* reg = 0.0001 , error = 0.89078
* reg = 1e-05 , error = 0.86266
* reg = 1e-06 , error = 0.88262
* reg = 1e-07 , error = 0.88934

### learning_rate = 1e-07

* reg = 100000.0 , error = 0.90514
* reg = 10000.0 , error = 0.89092
* reg = 1000.0 , error = 0.90092003
* reg = 100.0 , error = 0.89354
* reg = 10.0 , error = 0.89842
* reg = 1 , error = 0.89966
* reg = 0.1 , error = 0.89522
* reg = 0.01 , error = 0.90982
* reg = 0.001 , error = 0.901
* reg = 0.0001 , error = 0.88848
* reg = 1e-05 , error = 0.89658
* reg = 1e-06 , error = 0.90216
* reg = 1e-07 , error = 0.88854

### learning_rate = 1e-08

* reg = 100000.0 , error = 0.90786
* reg = 10000.0 , error = 0.90202
* reg = 1000.0 , error = 0.90884
* reg = 100.0 , error = 0.89832
* reg = 10.0 , error = 0.90188
* reg = 1 , error = 0.884
* reg = 0.1 , error = 0.89996
* reg = 0.01 , error = 0.89356
* reg = 0.001 , error = 0.89818
* reg = 0.0001 , error = 0.8985
* reg = 1e-05 , error = 0.90462
* reg = 1e-06 , error = 0.90929997
* reg = 1e-07 , error = 0.89192

## second try

* learning_rates = [9e-2,7e-2,5e-2,3e-2,1e-2,9e-3,7e-3,5e-3,3e-3,1e-3]
* regularization_strengths = [1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

### learning_rate = 0.09

* reg = 100.0 , error = 0.89794
* reg = 10.0 , error = 0.90032
* reg = 1 , error = 0.88878
* reg = 0.1 , error = 0.80676
* reg = 0.01 , error = 0.65616
* reg = 0.001 , error = 0.67326
* reg = 0.0001 , error = 0.89862
* reg = 1e-05 , error = 0.89794
* reg = 1e-06 , error = 0.90032
* reg = 1e-07 , error = 0.89862

### learning_rate = 0.07

* reg = 100.0 , error = 0.88878
* reg = 10.0 , error = 0.90142
* reg = 1 , error = 0.82992
* reg = 0.1 , error = 0.54664004
* reg = 0.01 , error = 0.57904
* reg = 0.001 , error = 0.79426
* reg = 0.0001 , error = 0.62567997
* reg = 1e-05 , error = 0.55464
* reg = 1e-06 , error = 0.90032
* reg = 1e-07 , error = 0.54526

### learning_rate = 0.05

* reg = 100.0 , error = 0.88878
* reg = 10.0 , error = 0.49914002
* reg = 1 , error = 0.89862
* reg = 0.1 , error = 0.50886
* reg = 0.01 , error = 0.4989
* reg = 0.001 , error = 0.51688004
* reg = 0.0001 , error = 0.48115999
* reg = 1e-05 , error = 0.5061
* reg = 1e-06 , error = 0.48610002
* reg = 1e-07 , error = 0.54886

### learning_rate = 0.03

* reg = 100.0 , error = 0.49582
* reg = 10.0 , error = 0.42894
* reg = 1 , error = 0.4537
* reg = 0.1 , error = 0.44412
* reg = 0.01 , error = 0.45599997
* reg = 0.001 , error = 0.4561
* reg = 0.0001 , error = 0.47746003
* reg = 1e-05 , error = 0.44088
* reg = 1e-06 , error = 0.46528
* reg = 1e-07 , error = 0.46296

### learning_rate = 0.01

* reg = 100.0 , error = 0.34781998
* reg = 10.0 , error = 0.34583998
* reg = 1 , error = 0.34818
* reg = 0.1 , error = 0.36104
* reg = 0.01 , error = 0.35486
* reg = 0.001 , error = 0.35556
* reg = 0.0001 , error = 0.35688
* reg = 1e-05 , error = 0.35698003
* reg = 1e-06 , error = 0.36567998
* reg = 1e-07 , error = 0.35138

### learning_rate = 0.009

* reg = 100.0 , error = 0.33946002
* reg = 10.0 , error = 0.3563
* reg = 1 , error = 0.35303998
* reg = 0.1 , error = 0.35284
* reg = 0.01 , error = 0.35454
* reg = 0.001 , error = 0.3516
* reg = 0.0001 , error = 0.35322
* reg = 1e-05 , error = 0.35184002
* reg = 1e-06 , error = 0.35070002
* reg = 1e-07 , error = 0.35355997

### learning_rate = 0.007

* reg = 100.0 , error = 0.33446002
* reg = 10.0 , error = 0.33062
* reg = 1 , error = 0.3505
* reg = 0.1 , error = 0.33804
* reg = 0.01 , error = 0.34855998
* reg = 0.001 , error = 0.33786
* reg = 0.0001 , error = 0.35167998
* reg = 1e-05 , error = 0.35268003
* reg = 1e-06 , error = 0.33995998
* reg = 1e-07 , error = 0.35442

### learning_rate = 0.005

* reg = 100.0 , error = 0.34425998
* reg = 10.0 , error = 0.33374
* reg = 1 , error = 0.33932
* reg = 0.1 , error = 0.34174
* reg = 0.01 , error = 0.34127998
* reg = 0.001 , error = 0.33951998
* reg = 0.0001 , error = 0.34666002
* reg = 1e-05 , error = 0.34890002
* reg = 1e-06 , error = 0.34083998
* reg = 1e-07 , error = 0.33944

### learning_rate = 0.003

* reg = 100.0 , error = 0.31300002
* reg = 10.0 , error = 0.3315
* reg = 1 , error = 0.33187997
* reg = 0.1 , error = 0.34556
* reg = 0.01 , error = 0.34324002
* reg = 0.001 , error = 0.3455
* reg = 0.0001 , error = 0.34484
* reg = 1e-05 , error = 0.3337
* reg = 1e-06 , error = 0.33683997
* reg = 1e-07 , error = 0.33674002

### learning_rate = 0.001

* reg = 100.0 , error = 0.32972002
* reg = 10.0 , error = 0.3255
* reg = 1 , error = 0.32856
* reg = 0.1 , error = 0.33683997
* reg = 0.01 , error = 0.34908003
* reg = 0.001 , error = 0.32647997
* reg = 0.0001 , error = 0.33569998
* reg = 1e-05 , error = 0.33678
* reg = 1e-06 , error = 0.32661998
* reg = 1e-07 , error = 0.32392


